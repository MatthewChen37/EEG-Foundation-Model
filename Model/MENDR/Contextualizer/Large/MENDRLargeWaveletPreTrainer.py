import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRWaveletContextualizer
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian
import mlflow
import matplotlib.pyplot as plt
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRLargeWaveletPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, wavelet_contextualizer, optimizer, cfg, **kwargs):
		self.svd = SVD.apply
		self.contrastive_loss_fn = nn.CrossEntropyLoss()

		# Freeze the autoencoder and disable the decoder
		for param in autoencoder.parameters():
			param.requires_grad = False
		autoencoder.eval()
		for band, encoder_decoder in autoencoder.encoder_decoders.items():
			encoder_decoder.disableDecoder()

		assert isinstance(wavelet_contextualizer, MENDRWaveletContextualizer), f"Contextualizer must be of type MENDRWaveletContextualizer, but got {type(wavelet_contextualizer)}"

		super(MENDRLargeWaveletPreTrainer, self).__init__(
			autoencoder=autoencoder,
		 	wavelet_contextualizer=wavelet_contextualizer,
			optimizer=optimizer,
			cfg=cfg,
			**kwargs)

		# Clamp gradients
		# This clips gradients before backpropagation: https://stackoverflow.com/a/54816498
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -cfg.training_params.gradient_clip_value, cfg.training_params.gradient_clip_value))

	def forward(self, data):
		if self.train_mode == "wavelet":
			patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)
			w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'wavelet_manifold_output': wavelet_manifold_output,
					'wavelet_loss': w_loss,
					'wavelet_acc': w_correct / w_pairs,
					}
		elif self.train_mode == "combined":
			patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = self.epochMaskedRecon(wavelet_manifold_output,
																															epoched_shape,
																															self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'wavelet_manifold_output': wavelet_manifold_output,
				}
		else:
			raise ValueError(f"Invalid training mode: {self.train_mode}. Must be 'wavelet' or 'combined'.")
		
	def leave_one_out(self, embeddings, criterion, negatives=20):
		"""
		Compute leave-one-out loss for wavelet embeddings.

		Args:
			embeddings (dict): Dictionary of SPD wavelet embeddings.
			criterion: Loss function (e.g., CrossEntropyLoss).
			negatives: Number of Negatives will be negatives - 1 (more like a sub-batch selection)

		Returns:
			loss (torch.Tensor): Total leave-one-out loss.
			correct (int): Number of correct predictions.
			pairs (int): Number of prediction pairs.
		"""
		frequency_bands = list(embeddings.keys())
		num_targets = len(frequency_bands)
		batch_size = embeddings['delta'].shape[0]

		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			# Average embeddings of all other modalities
			other_embeddings = []
			negative_selection = torch.randperm(batch_size)
			negative_indices = negative_selection[:negatives]
			for j in list(range(i)) + list(range(i + 1, num_targets)):
				embedding_tensor = embeddings[frequency_bands[j]].clone().view(-1, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out) # [Batch * epochs, C, C]
				embedding_tensor = embedding_tensor[negative_indices]
				other_embeddings.append(embedding_tensor)

			curr_target = embeddings[frequency_bands[i]].clone().view(-1, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out)[negative_indices]


			other_embeddings_mean = self.mendr_model.mendr_contextualizer.WaveletContextualizer._batch_LogEuclideanMean(other_embeddings, frequency_bands[i])

			# Why does this fail for higher precisions?
			# Answer: AttenionManifold's forward and SPDTransforms Forward are numerically unstable for FloatingPoint Precision calculations
			# I.E. it will create attention values that are approximately symmetric matrices up to a certain precision.
			# If we change the tensor's values to Double rather than float this assertion passes for PRECISION=16
			# However, to accelerate training at scale we need to keep everything as Floats
			# Note that the total number of digits a Float32 can store is around 6 to 9:
			# https://stackoverflow.com/questions/56514892/how-many-digits-can-float8-float16-float32-float64-and-float128-contain
			# assuming torch.float = np.float32
			# Potential future direction is "Quantizing" SPD matrices
			# assert torch.allclose(curr_target, curr_target.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(curr_target)
			# assert torch.allclose(other_embeddings_mean, other_embeddings_mean.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(other_embeddings_mean)

			# Compute logits
			logits = self._batchWiseMatrixSimilarity(curr_target, other_embeddings_mean)
			labels = torch.arange(logits.shape[0], device=self.device)

			# Forward loss
			forward_logits = logits
			l = criterion(forward_logits, labels)
			loss += l
			correct += (torch.argmax(forward_logits, axis=0) == labels).sum().item()
			pairs += forward_logits.size(0)

			# Reverse loss - Ensure logits are symmetric
			reverse_logits = logits.T
			l = criterion(reverse_logits, labels)
			loss += l
			correct += (torch.argmax(reverse_logits, axis=0) == labels).sum().item()
			pairs += reverse_logits.size(0)
		return loss, correct, pairs

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B, eps=1e-12):
		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		# Based on the Log-Euclidean metric
		a_u, a_s, a_v = self.svd(batch_A + eps)
		b_u, b_s, b_v = self.svd(batch_B + eps)
		tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s + eps)) @ a_v.permute(0, 2, 1)
		tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s + eps)) @ b_v.permute(0, 2, 1)
		inner_term = tensor_log_A[:, None, :, :] - tensor_log_B[None, :, :, :]
		output = torch.linalg.matrix_norm(inner_term, ord='fro')

		output = 1 / (1 + torch.log(1 + output))
		output = output * torch.exp(self.wavelet_contextualizer.temp1)
		return output