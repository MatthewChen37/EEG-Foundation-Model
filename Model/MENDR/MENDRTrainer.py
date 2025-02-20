import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import ptwt
from ..baseModelTrainer import BaseModelTrainer
from ..transforms import RandomTemporalCrop, RandomGaussianNoise, RandomFTSurrogate
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch
from .safeSVD import SVD, svdv2

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
PRECISION = 9 # Number of decimal places to consider equal

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, encoder, contextualizer, config, **kwargs):
		# Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True))
		self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
		self.contrastive_loss_fn_combined = nn.MSELoss()
		self.negatives_loo = 50

		# Mask is a learnable SPD matrix
		self.mask = np.random.rand(contextualizer.channels, contextualizer.channels)
		self.mask = torch.from_numpy(np.dot(self.mask, self.mask.transpose()))
		self.mask = nn.Parameter(self.mask, requires_grad=True)

		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, 
			temp1=self.temp1, mask=self.mask, contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
			contrastive_loss_fn_combined = self.contrastive_loss_fn_combined, lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), ckpt_dir=config.ckpt_dir, **kwargs)

		self.svd = SVD.apply

		''' Unused
		self.RandomGaussianNoise = RandomGaussianNoise()
		self.RandomFTSurrogate = RandomFTSurrogate(phase_noise_magnitude=0.2, random_state=config.random_state)
		'''	
	def forward(self, data):
		relevant_bands = [data[band] for band in BANDS]

		inputs = dict(zip(BANDS, relevant_bands))

		encoder_output = self.encoder(data['graph'], inputs)

		wavelet_manifold_output, epoched_shape = self.WaveletContextualizer(encoder_output)

		# Wavelet wise contrastive loss, i.e. Multi-Resolution loss
		w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)

		# Combined contrastive loss
		riemannian_loss = self.epochMaskedRecon(wavelet_manifold_output, epoched_shape, self.contrastive_loss_fn_combined)

		return {
				'encoder_output': encoder_output,
				'combined_r2e_output': combined_r2e_output,
				'combined_manifold_output': combined_manifold_output,
				'wavelet_manifold_output': wavelet_manifold_output,
				'wavelet_loss': w_loss,
				'euclidean_loss': euclidean_loss,
				'riemannian_loss': riemannian_loss,
				'wavelet_acc': w_correct / w_pairs
		}
	
	def reconstruction_loss(self, inputs, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def calculate_metrics(self, combined_total_loss, combined_euclidean_loss, combined_riemannian_loss, wavelet_loss, wavelet_acc, recon_loss):
		return {
			'Combined Total Loss': combined_total_loss,
			'Combined Euclidean Loss': combined_euclidean_loss,
			'Combined Riemannian Loss': combined_riemannian_loss,
			'Wavelet Loss': wavelet_loss,
			'Wavelet Acc': wavelet_acc,
			'Recon Loss': recon_loss,
		}

	def epochMaskedRecon(self, wavelet_manifold_output, epoched_shape, criterion):
		frequency_bands = list(wavelet_manifold_output.keys())
		batch_size = epoched_shape[0]
		with torch.no_grad(): # Don't need gradients for random indices
			masked_epochs = torch.randint(self.contextualizer.epochs, (batch_size,))

		wavelet_manifold_output_masked = dict()
		for band, spd_batch in wavelet_manifold_output.items():
			#wavelet_manifold_output[band] = wavelet_manifold_output[band].view(epoched_shape[0], epoched_shape[1], spd_batch.shape[1], spd_batch.shape[2])
			wavelet_manifold_output_masked[band] = spd_batch.clone()
			wavelet_manifold_output_masked[band] = wavelet_manifold_output_masked[band].view(epoched_shape[0], epoched_shape[1], spd_batch.shape[1], spd_batch.shape[2])

		# We mask one epoch calculate the LEM and then compare it with the full LEM
		# [B, E, C, C]
		for batch_idx, masked_epoch_idx in enumerate(masked_epochs):
			for band in wavelet_manifold_output_masked.keys():
				wavelet_manifold_output_masked[band][batch_idx, masked_epoch_idx, :, :] = self.mask

		for band in wavelet_manifold_output_masked.keys():
			wavelet_manifold_output_masked[band] = wavelet_manifold_output_masked[band].view(epoched_shape[0] * epoched_shape[1], spd_batch.shape[1], spd_batch.shape[2])

		
		_, combined_manifold_output_masked = self.contextualizer.CombinedContextualizer(wavelet_manifold_output_masked, epoched_shape)

		# Log Euclidean Mean
		combined_manifold_output = torch.stack(list(wavelet_manifold_output.values()), dim=1)
		combined_manifold_output = self.contextualizer.CombinedContextualizer.combined_attention.tensor_log(combined_manifold_output)
		combined_manifold_output = combined_manifold_output.sum(dim=1, keepdim=True) / combined_manifold_output.shape[1]
		combined_manifold_output = self.contextualizer.CombinedContextualizer.combined_attention.tensor_exp(combined_manifold_output)
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		combined_manifold_output_masked = combined_manifold_output_masked.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		combined_manifold_output_important_part = torch.empty(epoched_shape[0], combined_manifold_output.shape[2], combined_manifold_output.shape[3]).to(self.device)
		combined_manifold_output_masked_important_part = torch.empty(epoched_shape[0], combined_manifold_output.shape[2], combined_manifold_output.shape[3]).to(self.device)

		for batch_idx, masked_epoch_idx in enumerate(masked_epochs):
			combined_manifold_output_important_part[batch_idx, :, :] = combined_manifold_output[batch_idx, masked_epoch_idx, :, :]
			combined_manifold_output_masked_important_part[batch_idx, :, :] = combined_manifold_output_masked[batch_idx, masked_epoch_idx, :, :]

		# Reconstruction loss
		riemannian_loss = criterion(combined_manifold_output_important_part, combined_manifold_output_masked_important_part)
		return riemannian_loss

	def leave_one_out(self, embeddings, criterion, negatives=50):
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

			# trace normalization
			'''
			trace = other_embeddings.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
			trace = trace.view(-1, 1, 1)
			other_embeddings /= trace
			identity = torch.eye(other_embeddings.shape[-1], other_embeddings.shape[-1], device=self.device).to(self.device).repeat(other_embeddings.shape[0], 1, 1)
			other_embeddings = other_embeddings + (1e5 * identity)
			'''
			assert torch.allclose(other_embeddings, other_embeddings.mT, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {other_embeddings}"

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
		return loss, correct, pairs

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


