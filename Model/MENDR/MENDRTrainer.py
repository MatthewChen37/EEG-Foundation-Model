import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import ptwt
from ..baseModelTrainer import BaseModelTrainer
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch
from .safeSVD import SVD

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
PRECISION = 7 # Number of decimal places to consider equal

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, encoder, contextualizer, r2e, config, **kwargs):
		if config.multi_gpu:
			stembedder = nn.DataParallel(stembedder)
			encoder = nn.DataParallel(encoder)
			contextualizer = nn.DataParallel(contextualizer)
			r2e = nn.DataParallel(r2e)
			decoder = nn.DataParallel(decoder)
		
		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, r2e=r2e, 
			contrastive_loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
			metrics=dict(), 
			save_model_directory=config.save_model_directory, **kwargs)
		
		# Initialize temperature as a trainable parameter
		self.temp = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True))
		# TODO: Fix these later...
		self.alpha = config.enc_feat_l2
		self.band_coeffs = {
			'delta': 0,
			'theta': 1,
			'alpha': 1,
			'beta': 0,
			'gamma': 0,
		}
		self.num_negatives = config.num_negatives
		self.svd = SVD.apply

	def forward(self, data):
		relevant_bands = [data[band].float().to(self.device) for band in BANDS]
		inputs = dict(zip(BANDS, relevant_bands))
		encoder_output = self.encoder(data['graph'], inputs)
		contextualizer_output, shape, wavelet_embeddings = self.contextualizer(encoder_output)
		loss, correct, pairs = self.leave_one_out(wavelet_embeddings, self.contrastive_loss_fn)
		euclidean_embeddings = self.r2e(contextualizer_output, shape)
		return contextualizer_output, shape, loss, encoder_output, correct, pairs, euclidean_embeddings
	
	def calculate_loss(self, inputs, encoder_decoder_output, contrastive_loss):
		recon_loss = self._reconstruction_loss(inputs, encoder_decoder_output)
		return contrastive_loss + recon_loss, recon_loss

	def _reconstruction_loss(self, inputs, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def calculate_metrics(self, correct, pairs, contrastive_loss, recon_loss):
		return {
			'Contrastive Accuracy': correct / pairs,
			'Contrastive Loss': contrastive_loss.item(),
			'Reconstruction Loss': recon_loss.item()
		}
	
	def leave_one_out(self, embeddings, criterion):
		"""
		Compute leave-one-out loss for wavelet embeddings.

		Args:
			embeddings (dict): Dictionary of SPD wavelet embeddings.
			criterion: Loss function (e.g., CrossEntropyLoss).
			temperature (torch.nn.Parameter): Temperature parameter for scaling logits.

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
				embedding_tensor = embeddings[frequency_bands[j]][0]
				original_batch_shape = embeddings[frequency_bands[j]][1] #TODO: Is this never used?
				other_embeddings.append(embedding_tensor)
			other_embeddings = torch.stack(other_embeddings).sum(0) / (num_targets - 1)

			# trace normalization
			curr_target = embeddings[frequency_bands[i]][0] # curr_target already trace normalized
			
			trace = other_embeddings.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
			trace = trace.view(-1, 1, 1)
			other_embeddings /= trace
			identity = torch.eye(other_embeddings.shape[-1], other_embeddings.shape[-1], device=self.device).to(self.device).repeat(other_embeddings.shape[0], 1, 1)
			other_embeddings = other_embeddings + (1e-5 * other_embeddings)
			assert torch.allclose(other_embeddings, other_embeddings.mT, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {other_embeddings}"

			# Compute logits
			logits = self._batchWiseMatrixSimilarity(curr_target, other_embeddings) 
			labels = torch.arange(logits.shape[0], device=self.device)

			# Forward loss
			l = criterion(logits, labels)
			loss += l
			correct += (torch.argmax(logits, axis=0) == labels).sum().item()
			pairs += logits.size(0)

			# Reverse loss
			l = criterion(logits.T, labels)
			loss += l
			correct += (torch.argmax(logits, axis=1) == labels).sum().item()
			pairs += logits.size(0)
		return loss, correct, pairs

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B):
		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		for i in range(batch_A.shape[0]):
			for j in range(batch_B.shape[0]):
				# Based on the Log-Euclidean metric 
				a_u, a_s, a_v = self.svd(batch_A[i, :, :])
				b_u, b_s, b_v = self.svd(batch_B[j, :, :])
				tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s)) @ a_v.permute(1, 0)
				tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s)) @ b_v.permute(1, 0)

				inner_term = tensor_log_A - tensor_log_B
				output[i, j] = torch.linalg.matrix_norm(inner_term, ord='fro') * torch.exp(self.temp)

		return output


