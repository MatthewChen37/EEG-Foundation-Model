import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from ..baseModelTrainer import BaseModelTrainer
from ..transforms import RandomTemporalCrop, RandomGaussianNoise, RandomFTSurrogate
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch
from .mAtt import StiefelParameter
from .safeSVD import SVD, svdv2
from scipy.linalg import orth
from .MENDRContextualizerLarge import MENDRContextualizerLarge
from .MENDRContextualizerTiny import MENDRContextualizerTiny
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotReconstruction import plotReconstruction
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian, plotWaveletEmbeddingsEuclidean
import mlflow

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ABS_PRECISION = 3 # Number of decimal places to consider equal
REL_PRECISION = 1 # Relative tolerance precision
class MENDRPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, MENDR, config, **kwargs):
		self.negatives_loo = 20
		self.mask_ratio = 0.5
		self.svd = SVD.apply
		self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
		self.contrastive_loss_fn_combined = nn.MSELoss()
		self.pref_vector = None
		self.aggregator = UPGrad(self.pref_vector) # TODO: include pref_vector

		super(MENDRPreTrainer, self).__init__(mendr_model=MENDR, contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
			contrastive_loss_fn_combined = self.contrastive_loss_fn_combined, aggregator=self.aggregator, lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), ckpt_dir=config.ckpt_dir, **kwargs)

		# Clamp gradients
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -1e7, 1e7))

		''' Unused
		self.RandomGaussianNoise = RandomGaussianNoise()
		self.RandomFTSurrogate = RandomFTSurrogate(phase_noise_magnitude=0.2, random_state=config.random_state)
		'''	
	def forward(self, data):
		'''
		Looks similar to MENDR_model forward
		but is modified for the contrastive learning task(s)
		'''
		graphs = data['graph']
		patchified_inputs = self.mendr_model._super_patchify(data)
		encodings, decodings = self.mendr_model.mendr_encoder(graphs, patchified_inputs)
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)

			# Wavelet wise contrastive loss, i.e. Multi-Resolution loss
			w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)

			# Combined contrastive loss
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked = self.epochMaskedRecon(wavelet_manifold_output, epoched_shape, self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'combined_manifold_output_masked': combined_manifold_output_masked,
					'riemannian_loss': riemannian_loss,
					'wavelet_manifold_output': wavelet_manifold_output,
					'wavelet_loss': w_loss,
					'wavelet_acc': w_correct / w_pairs
			}
		elif self.mendr_model.contextualizer_size.upper() == "TINY":
			batch_size = patchified_inputs['delta'].shape[0]
			patches = patchified_inputs['delta'].shape[1]
			# Combined contrastive loss
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked = self.epochMaskedReconTiny(encodings, self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'combined_manifold_output_masked': combined_manifold_output_masked,
					'riemannian_loss': riemannian_loss,
			}
		else:
			raise ValueError("Unidentified Contextualizer Type")

	def backward(self, shared_features, contrastive_losses, reconstruction_losses):
		self.optimizer.zero_grad()
		if self.pref_vector != None:
			assert len(contrastive_losses) + len(reconstruction_losses) == len((self.pref_vector)), f"Contrastive Losses: {len(contrastive_losses)} Reconstruction Losses: {len(reconstruction_losses)} Pref Vector: {len(self.pref_vector)}"
		losses = contrastive_losses + list(reconstruction_losses.values())
		mtl_backward(losses=losses, features=shared_features, aggregator=self.aggregator, retain_graph=False)

		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			# Only large model has temp parameter, which is used in wavelet loss
			# Clamp temperature to non-negative values
			with torch.no_grad():
				self.mendr_model.mendr_contextualizer.temp1.copy_(torch.clamp(self.mendr_model.mendr_contextualizer.temp1, min=0.0))

	def train_step(self, inputs):
		self.train(True)
		outputs = self.forward(inputs)
		recon_losses = WaveletReconstructionLoss(outputs['patchified_inputs'], outputs['decodings']) # Reconstruction Loss returns a dictionary
		shared_features = [encoding for band, encoding in outputs['encodings'].items()]
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			# In a future work, figure out a way to multi-level backpropagate (i.e. jacobians for the wavelet loss and riemannian loss on riemannian and wavelet loss)
			self.backward(shared_features=shared_features, contrastive_losses=[outputs['riemannian_loss'] + outputs['wavelet_loss']], reconstruction_losses=recon_losses)
			train_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), outputs['wavelet_loss'].item(), outputs['wavelet_acc'], recon_losses)
			self.optimizer.step()
		elif self.mendr_model.contextualizer_size.upper() == "TINY":
			self.backward(shared_features=shared_features, contrastive_losses=[outputs['riemannian_loss']], reconstruction_losses=recon_losses)
			train_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), None, None, recon_losses)
			self.optimizer.step()
		else:
			raise ValueError("Unidentified Contextualizer Type")
		train_metrics["lr"] = f"{self.optimizer.scheduler.get_last_lr()[0]:.7f}"

		for metric in train_metrics: # Don't want tensors in the metrics
			if isinstance(train_metrics[metric], torch.Tensor):
				train_metrics[metric] = train_metrics[metric].item()
		return train_metrics

	def evaluate_step(self, inputs, step_idx):
		self.train(False)
		with torch.no_grad():
			outputs = self.forward(inputs)
			recon_losses = WaveletReconstructionLoss(outputs['patchified_inputs'], outputs['decodings']) # Reconstruction Loss returns a dictionary
			if self.mendr_model.contextualizer_size.upper() == 'LARGE':
				eval_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), outputs['wavelet_loss'].item(), outputs['wavelet_acc'], recon_losses)
			elif self.mendr_model.contextualizer_size.upper() == 'TINY':
				eval_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), None, None, recon_losses)
			else:
				raise ValueError("Unidentified Contextualizer Type")

			if step_idx == 0: # Log only the first batch in the validation set
				wavelet_manifold_output = outputs['wavelet_manifold_output']
				combined_manifold_output = outputs['combined_manifold_output']
				combined_manifold_output_masked = outputs['combined_manifold_output_masked']
				batch_size = combined_manifold_output.shape[0]
				num_patches = combined_manifold_output.shape[1]
				combined_manifold_output = combined_manifold_output.reshape(wavelet_manifold_output['delta'].shape)
				combined_manifold_output_masked = combined_manifold_output_masked.reshape(wavelet_manifold_output['delta'].shape)
				assert len(inputs['subject_name']) == batch_size, f"Subject Name Length: {len(inputs['subject_name'])} Batch Size: {batch_size}"
				wavelet_figs, combined_fig = plotSPDEmbedding(wavelet_manifold_output, combined_manifold_output, combined_manifold_output_masked, inputs['subject_name'])
				for band, wavelet_fig in wavelet_figs.items():
					mlflow.log_figure(wavelet_fig, f"epoch_{self.epoch}_{band}_wavelet_embeddings.png")
					plt.close(wavelet_fig)
				mlflow.log_figure(combined_fig, f"epoch_{self.epoch}_combined_embeddings.png")
				plt.close(combined_fig)

				fig = plotReconstruction(input_batch, outputs['decodings'], f"epoch_{self.epoch}")
				mlflow.log_figure(fig, f"epoch_{self.epoch}_reconstruction.png")
				plt.close(fig)

			return eval_metrics

	def _calculate_metrics(self, combined_riemannian_loss, wavelet_loss, wavelet_acc, recon_loss):
		if wavelet_loss is None or wavelet_acc is None:
			return {
				'Combined Riemannian Loss': combined_riemannian_loss,
			} | recon_loss

		else:
			return {
				'Combined Riemannian Loss': combined_riemannian_loss,
				'Wavelet Loss': wavelet_loss,
				'Wavelet Acc': wavelet_acc,
			} | recon_loss

	# Useful for debugging, only called on assertion error
	def _findNonSymmetry(self, A):
		torch.set_printoptions(precision=ABS_PRECISION + REL_PRECISION, threshold=200, linewidth=1e2)
		if len(A.shape) == 2:
			non_symmetries = torch.where(A != A.T)
			output = f"Indices: {non_symmetries} OgVals: {A[non_symmetries]} TransposeVals: {A.T[non_symmetries]}, Num_indices: {non_symmetries[0].shape}"
		else: # Batch of matrices
			non_symmetries = torch.where(A != A.mT)
			output = f"Indices: {non_symmetries} OgVals: {A[non_symmetries]} TransposeVals: {A.mT[non_symmetries]}, Num_indices: {non_symmetries[0].shape}"
		return output

	def epochMaskedRecon(self, wavelet_manifold_output, epoched_shape, criterion):
		frequency_bands = list(wavelet_manifold_output.keys())
		batch_size = epoched_shape[0]
		num_epochs = epoched_shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output_masked, mask_idxes = self.mendr_model.mendr_contextualizer.CombinedContextualizer(wavelet_manifold_output, epoched_shape, mask_ratio=self.mask_ratio)
		# Log Euclidean Mean True 
		combined_manifold_output = self.mendr_model.mendr_contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(wavelet_manifold_output)

		combined_manifold_output_masked = combined_manifold_output_masked.view(epoched_shape[0], epoched_shape[1], combined_manifold_output_masked.shape[1], combined_manifold_output_masked.shape[2])
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])

		return riemannian_loss, combined_manifold_output, combined_manifold_output_masked

	def epochMaskedReconTiny(self, wavelet_manifold_output, criterion):
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output_masked, combined_manifold_output, mask_idxes = self.mendr_model.mendr_contextualizer(wavelet_manifold_output, mask_ratio=self.mask_ratio)

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])

		return riemannian_loss, combined_manifold_output, combined_manifold_output_masked

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

			with torch.no_grad(): # Gradients don't need to be calculated for indices
				negative_selection = torch.randperm(batch_size)
				negative_indices = negative_selection[:negatives]

			for j in list(range(i)) + list(range(i + 1, num_targets)):
				embedding_tensor = embeddings[frequency_bands[j]] # [Batch * epochs, C, C]
				embedding_tensor = embedding_tensor[negative_indices]
				other_embeddings.append(embedding_tensor)

			curr_target = embeddings[frequency_bands[i]][negative_indices]
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
			#print(curr_target.shape, other_embeddings.shape)
			#logits = torch.matmul(curr_target, other_embeddings.T) * torch.exp(self.temp1)
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
		return 2.5e-4*loss, correct, pairs

	'''
	Currently not being used
	'''
	def check_degenerate_singular_values(self, matrix, tol=1e-5):
		"""
		Checks for degenerate singular values in a matrix.

		Args:
			matrix (torch.Tensor): The input matrix.
			tol (float): Tolerance for considering singular values as degenerate.

		Returns:
			bool: True if degenerate singular values are found, False otherwise.
		"""

		singular_values = torch.linalg.svdvals(matrix)
		diff = torch.abs(singular_values[:, :-1] - singular_values[:, 1:])
		return torch.any(diff < tol), torch.where(diff < tol)

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B):
		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		# Based on the Log-Euclidean metric
		'''
		batch_A_rank = torch.linalg.matrix_rank(batch_A)
		batch_B_rank = torch.linalg.matrix_rank(batch_B)
		if (batch_A_rank != batch_A.shape[1]).any():
			raise Exception("batch A not full rank")
		if (batch_B_rank != batch_B.shape[1]).any():
			raise Exception("batch B not full rank")

		badA, whereA = self.check_degenerate_singular_values(batch_A)
		print(whereA[0].shape, whereA[1].shape, torch.unique(whereA[0]).shape)

		if badA:
			print(whereA)
			raise Exception("batch A degenerate")

		if self.check_degenerate_singular_values(batch_B):
			raise Exception("batch B degenerate")
		
		a_u = torch.zeros(batch_A.shape).to(self.device)
		a_s = torch.zeros(batch_A.shape[0], batch_A.shape[1]).to(self.device)
		a_v = torch.zeros(batch_A.shape).to(self.device)

		b_u = torch.zeros(batch_B.shape).to(self.device)
		b_s = torch.zeros(batch_B.shape[0], batch_B.shape[1]).to(self.device)
		b_v = torch.zeros(batch_B.shape).to(self.device)

		for i in range(batch_A.shape[0]):
			a_u[i], a_s[i], a_v[i] = self.svd(batch_A[i])
			b_u[i], b_s[i], b_v[i] = self.svd(batch_B[i])

		'''
		a_u, a_s, a_v = self.svd(batch_A)
		b_u, b_s, b_v = self.svd(batch_B)

		tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s)) @ a_v.permute(0, 2, 1)
		tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s)) @ b_v.permute(0, 2, 1)
		inner_term = tensor_log_A[:, None, ...] - tensor_log_B[None, ...]
		output = torch.linalg.matrix_norm(inner_term, ord='fro') * torch.exp(self.mendr_model.mendr_contextualizer.temp1)
		return output