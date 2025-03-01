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

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ABS_PRECISION = 3 # Number of decimal places to consider equal
REL_PRECISION = 1 # Relative tolerance precision
class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, MENDR, config, **kwargs):
		self.negatives_loo = 20
		self.svd = SVD.apply

		super(MENDRTrainer, self).__init__(mendr_model=MENDR, contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
			contrastive_loss_fn_combined = self.contrastive_loss_fn_combined, lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), ckpt_dir=config.ckpt_dir, **kwargs)

		''' Unused
		self.RandomGaussianNoise = RandomGaussianNoise()
		self.RandomFTSurrogate = RandomFTSurrogate(phase_noise_magnitude=0.2, random_state=config.random_state)
		'''	
	def forward(self, data):
		encoder_output, wavelet_manifold_output = self.mendr_model(data)
		
		# Wavelet wise contrastive loss, i.e. Multi-Resolution loss
		w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)

		# Combined contrastive loss
		riemannian_loss, combined_manifold_output, combined_manifold_output_masked, masked_epochs = self.epochMaskedRecon(wavelet_manifold_output, epoched_shape, self.contrastive_loss_fn_combined)

		return {
				'encoder_output': encoder_output,
				'combined_manifold_output': combined_manifold_output,
				'combined_manifold_output_masked': combined_manifold_output_masked,
				'riemannian_loss': riemannian_loss,
				'wavelet_manifold_output': wavelet_manifold_output,
				'wavelet_loss': w_loss,
				'wavelet_acc': w_correct / w_pairs
		}
	
	def reconstruction_loss(self, inputs, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def calculate_metrics(self, total_loss, combined_riemannian_loss, wavelet_loss, wavelet_acc, recon_loss):
		return {
			'Total Loss': total_loss,
			'Combined Riemannian Loss': combined_riemannian_loss,
			'Wavelet Loss': wavelet_loss,
			'Wavelet Acc': wavelet_acc,
			'Recon Loss': recon_loss,
		}

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
		with torch.no_grad(): # Don't need gradients for random indices
			masked_epochs = torch.randint(num_epochs, (batch_size,))

		combined_manifold_output_masked = self.contextualizer.CombinedContextualizer(wavelet_manifold_output, epoched_shape, mask=masked_epochs)

		# Log Euclidean Mean True 
		combined_manifold_output = self.contextualizer.CombinedContextualizer(wavelet_manifold_output, epoched_shape)
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		# Reconstruction loss
		riemannian_loss = criterion(combined_manifold_output_important_part, combined_manifold_output_masked_important_part)

		return 3e7*riemannian_loss, combined_manifold_output, combined_manifold_output_masked, masked_epochs

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
			other_embeddings = torch.stack(other_embeddings, dim=1)

			# Log Euclidean Mean - Tensor log shouldn't really be tied to the instanttiation of the object...
			other_embeddings_log = self.contextualizer.WaveletContextualizer.wavelet_attention_manifolds[frequency_bands[i]].tensor_log(other_embeddings)
			other_embeddings_mean = self.contextualizer.WaveletContextualizer.wavelet_attention_manifolds[frequency_bands[i]].tensor_exp(other_embeddings_log.sum(dim=1, keepdim=True) / other_embeddings_log.shape[1])[:, 0, :, :]

			# Why does this fail for higher precisions?
			# Answer: AttenionManifold's forward and SPDTransforms Forward are numerically unstable for FloatingPoint Precision calculations
			# I.E. it will create attention values that are approximately symmetric matrices up to a certain precision.
			# If we change the tensor's values to Double rather than float this assertion passes for PRECISION=16
			# However, to accelerate training at scale we need to keep everything as Floats
			# Note that the total number of digits a Float32 can store is around 6 to 9:
			# https://stackoverflow.com/questions/56514892/how-many-digits-can-float8-float16-float32-float64-and-float128-contain
			# assuming torch.float = np.float32
			# Potential future direction is "Quantizing" SPD matrices
			assert torch.allclose(curr_target, curr_target.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(curr_target)
			assert torch.allclose(other_embeddings_mean, other_embeddings_mean.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(other_embeddings_mean)

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
		return 0.2 * loss, correct, pairs

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
		output = torch.linalg.matrix_norm(inner_term, ord='fro') * torch.exp(self.temp1)
		return output


