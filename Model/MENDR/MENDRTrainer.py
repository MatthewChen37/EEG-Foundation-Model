import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import ptwt
from ..baseModelTrainer import BaseModelTrainer
from ..transforms import RandomTemporalCrop, RandomGaussianNoise
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch
from .safeSVD import SVD, svdv2

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
PRECISION = 7 # Number of decimal places to consider equal

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, encoder, contextualizer, config, **kwargs):
		# Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True))
		self.temp2 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True))
		if config.multi_gpu:
			encoder = nn.DataParallel(encoder)
			contextualizer = nn.DataParallel(contextualizer)
			self.temp1 = nn.DataParallel(self.temp1)
			self.temp2 = nn.DataParallel(self.temp2)
		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer,
			temp1=self.temp1, temp2=self.temp2, contrastive_loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), 
			save_model_directory=config.save_model_directory, **kwargs)

		self.svd = SVD.apply

		self.RandomGaussianNoise = RandomGaussianNoise()
	
	def forward(self, data):
		relevant_bands = [data[band] for band in BANDS]

		inputs = dict(zip(BANDS, relevant_bands))

		encoder_output = self.encoder(data['graph'], inputs)

		combined_r2e_output, combined_manifold_output, wavelet_r2e_output, wavelet_manifold_output = self.contextualizer(encoder_output)

		# Wavelet wise contrastive loss, i.e. Multi-headed loss
		w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_r2e_output, self.contrastive_loss_fn)

		# Combined contrastive loss
		c_loss, c_correct, c_pairs = self.simCLR(inputs, data['graph'], combined_r2e_output, self.contrastive_loss_fn)

		return {
				'encoder_output': encoder_output,
				'combined_r2e_output': combined_r2e_output,
				'combined_manifold_output': combined_manifold_output,
				'wavelet_r2e_output': wavelet_r2e_output,
				'wavelet_manifold_output': wavelet_manifold_output,
				'combined_loss': c_loss,
				'wavelet_loss': w_loss,
				'combined_acc': c_correct / c_pairs,
				'wavelet_acc': w_correct / w_pairs
		}
	
	def reconstruction_loss(self, inputs, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def calculate_metrics(self, combined_loss, wavelet_loss, recon_loss, combined_acc, wavelet_acc):
		return {
			'Combined Loss': combined_loss,
			'Wavelet Loss': wavelet_loss,
			'Recon Loss': recon_loss,
			'Combined Acc': combined_acc,
			'Wavelet Acc': wavelet_acc,
		}

	def simCLR(self, inputs, data_graph, combined_embedding, criterion):
		'''
		Batchwise Simple CLR Loss on combined attention euclidean embeddings.
		'''
		frequency_bands = list(inputs.keys())
		num_targets = inputs[frequency_bands[0]].shape[0]
		self.encoder.freeze_features(unfreeze=False)
		self.contextualizer.freeze_features(unfreeze=False)
		loss = 0.0
		correct = 0
		pairs = 0
		with torch.no_grad():
			transformed_inputs = dict()
			for band in frequency_bands:
				# Add random noise
				transformed_inputs[band] = self.RandomGaussianNoise(inputs[band], training=True)
			transformed_encoder_output = self.encoder(data_graph, transformed_inputs)
			transformed_embeddings, _, _, _ = self.contextualizer(transformed_encoder_output)
			# Compute logits
			logits = torch.matmul(combined_embedding, transformed_embeddings.T) * torch.exp(self.temp2)
			labels = torch.arange(combined_embedding.shape[0], device=self.device)

			# Forward loss
			forward_logits = logits
			l = criterion(forward_logits, labels)
			loss += l
			correct += (torch.argmax(forward_logits, axis=0) == labels).sum().item()
			pairs += forward_logits.size(0)

			# Reverse loss
			reverse_logits = logits.T
			l = criterion(reverse_logits, labels)
			loss += l
			correct += (torch.argmax(reverse_logits, axis=0) == labels).sum().item()
			pairs += reverse_logits.size(0)
		self.encoder.freeze_features(unfreeze=True)
		self.contextualizer.freeze_features(unfreeze=True)
		return loss, correct, pairs

	def leave_one_out(self, embeddings, criterion):
		"""
		Compute leave-one-out loss for wavelet embeddings.

		Args:
			embeddings (dict): Dictionary of SPD wavelet embeddings.
			criterion: Loss function (e.g., CrossEntropyLoss).

		Returns:
			loss (torch.Tensor): Total leave-one-out loss.
			correct (int): Number of correct predictions.
			pairs (int): Number of prediction pairs.
		"""
		frequency_bands = list(embeddings.keys())
		num_targets = len(frequency_bands)
		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			# Average embeddings of all other modalities
			other_embeddings = []
			for j in list(range(i)) + list(range(i + 1, num_targets)):
				embedding_tensor = embeddings[frequency_bands[j]]
				other_embeddings.append(embedding_tensor)
			other_embeddings = torch.stack(other_embeddings).sum(0) / (num_targets - 1)
			curr_target = embeddings[frequency_bands[i]]

			'''
			When embeddings where matrices, not vectors.
			# trace normalization
			trace = other_embeddings.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
			trace = trace.view(-1, 1, 1)
			other_embeddings /= trace
			identity = torch.eye(other_embeddings.shape[-1], other_embeddings.shape[-1], device=self.device).to(self.device).repeat(other_embeddings.shape[0], 1, 1)
			other_embeddings = other_embeddings + (1e5 * identity)
			assert torch.allclose(other_embeddings, other_embeddings.mT, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {other_embeddings}"
			'''
			# Compute logits
			# logits = self._batchWiseMatrixSimilarity(curr_target, other_embeddings)
			logits = torch.matmul(curr_target, other_embeddings.T) * torch.exp(self.temp1)
			labels = torch.arange(logits.shape[0], device=self.device)

			# Forward loss
			forward_logits = logits
			l = criterion(forward_logits, labels)
			loss += l
			correct += (torch.argmax(forward_logits, axis=0) == labels).sum().item()
			pairs += forward_logits.size(0)

			# Reverse loss
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

	'''
	Currently not being used
	'''
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
		output = torch.linalg.matrix_norm(inner_term, ord='fro') * 1

		return output


