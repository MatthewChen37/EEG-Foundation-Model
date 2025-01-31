import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import ptwt
from ..baseModelTrainer import BaseModelTrainer
from .MENDRContextualizer import _make_mask, _make_span_from_seeds
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
			metrics=dict(Accuracy=self._contrastive_accuracy), 
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
		return contextualizer_output, shape, loss, encoder_output
	
	def calculate_loss(self, inputs, encoder_decoder_output, contrastive_loss):
		recon_loss = self._reconstruction_loss(inputs, encoder_decoder_output)
		return contrastive_loss + recon_loss, recon_loss

	def _reconstruction_loss(self, inputs, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def _contrastive_accuracy(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		return self._simple_accuracy([labels], logits)
    
	def calculate_metrics(self, *inputs, outputs, recon_loss):
		"""
		Cosine Similarity from Calculating Similarity
		"""
		# "Logits" from _calculate_similarity
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		
		means = logits.mean(dim=0)
		return {
			'Negative Similarity': means[0].item(),
			'Positive Similarity': means[1].item(),
			'Contrastive Accuracy': self._simple_accuracy([labels], logits),
			'MASK_pct': self._mask_pct(inputs, outputs),
			'BENDR Reconstruction MSE': recon_loss.item()
		}
	
	@staticmethod
	def _simple_accuracy(inputs, outputs):
		if isinstance(outputs, (list, tuple)):
			outputs = outputs[0]
        # average over last dimensions
		while len(outputs.shape) >= 3:
			outputs = outputs.mean(dim=-1)
		return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()
	
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
			curr_target = embeddings[frequency_bands[i]][0]
			trace = curr_target.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
			trace = trace.view(-1, 1, 1)
			curr_target /= 0.5 * trace
			identity = torch.eye(curr_target.shape[-1], curr_target.shape[-1], device=self.device).to(self.device).repeat(curr_target.shape[0], 1, 1)
			curr_target = curr_target + (1e-5 * identity)

			trace = other_embeddings.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
			trace = trace.view(-1, 1, 1)
			other_embeddings /= 0.5 * trace
			identity = torch.eye(other_embeddings.shape[-1], other_embeddings.shape[-1], device=self.device).to(self.device).repeat(other_embeddings.shape[0], 1, 1)
			other_embeddings = other_embeddings + (1e-5 * other_embeddings)
			assert torch.allclose(other_embeddings, other_embeddings.mT, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {other_embeddings}"
			# Compute logits
			logits = self._batchWiseMatrixSimilarity(curr_target, other_embeddings, original_batch_shape) 
			labels = torch.arange(logits.shape[0], device=self.device)
			#labels = self._gen_labels(original_batch_shape).long()

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

	def _gen_labels(self, batch_shape):
		output = torch.zeros((batch_shape[0], batch_shape[1])).to(self.device)
		idx = 0
		for i in range(batch_shape[0]):
			for j in range(batch_shape[1]):
				output[i, j] = idx
				idx += 1
		return output

		
	def _batchWiseMatrixSimilarity(self, batch_A, batch_B, original_batch_shape):
		#a_u, a_s, a_v = torch.svd(batch_A)
		#b_u, b_s, b_v = torch.svd(batch_B)

		#print(a_s)

		#tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s)) @ a_v.permute(0, 2, 1)
		#tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s)) @ b_v.permute(0, 2, 1)

		#tensor_log_A = self.contextualizer.combined_attention.tensor_log(batch_A.view(original_batch_shape[0], original_batch_shape[1], batch_A.shape[1], batch_A.shape[2]))
		#tensor_log_B = self.contextualizer.combined_attention.tensor_log(batch_B.view(original_batch_shape[0], original_batch_shape[1], batch_B.shape[1], batch_B.shape[2]))

		#tensor_log_A = tensor_log_A.view(batch_A.shape)
		#tensor_log_B = tensor_log_B.view(batch_B.shape)

		#print(tensor_log_A.shape, tensor_log_B.shape)

		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		for i in range(batch_A.shape[0]):
			for j in range(batch_B.shape[0]):
				a_u, a_s, a_v = self.svd(batch_A[i, :, :])
				b_u, b_s, b_v = self.svd(batch_B[j, :, :])
				tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s)) @ a_v.permute(1, 0)
				tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s)) @ b_v.permute(1, 0)

				inner_term = tensor_log_A - tensor_log_B
				output[i, j] = torch.linalg.matrix_norm(inner_term, ord='fro') * torch.exp(self.temp)

		return output


