import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian
import mlflow
import matplotlib.pyplot as plt
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRLargePreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, contextualizer, cfg, **kwargs):
		self.mask_ratio = cfg.training_params.mask_ratio
		self.svd = SVD.apply
		self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
		self.contrastive_loss_fn_combined = nn.MSELoss()
		self.train_mode = None
	
		assert isinstance(contextualizer, MENDRContextualizerLarge), f"Contextualizer must be of type MENDRContextualizerLarge, but got {type(contextualizer)}"

		super(MENDRLargePreTrainer, self).__init__(autoencoder=autoencoder, contextualizer=contextualizer,
			contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
			contrastive_loss_fn_combined=self.contrastive_loss_fn,
			lr=cfg.train_params.learning_rate,
			l2_weight_decay=cfg.training_params.l2_weight_decay,
			metrics=dict(),
			ckpt_dir=cfg.training_params.ckpt_dir,
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
	
	def epochMaskedRecon(self, wavelet_manifold_output, epoched_shape, criterion):
		frequency_bands = list(wavelet_manifold_output.keys())
		batch_size = epoched_shape[0]
		num_epochs = epoched_shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output = self.mendr_model.mendr_contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(wavelet_manifold_output)
		combined_manifold_output_masked = combined_manifold_output.clone()
		combined_manifold_output_masked, mask_idxes = self.mendr_model.mendr_contextualizer.CombinedContextualizer(combined_manifold_output_masked, epoched_shape, mask_ratio=self.mask_ratio)
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])
		return self.contrastive_combined_loss_coeff * riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes



