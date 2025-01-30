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

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

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
		if config.encoder_grad_frac < 1:
            # TODO: I hope this works...
			for band, encoder_decoder in encoder.encoder_decoders.items():
				encoder_decoder.patch_embedder.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig 
						  if ig is not None else None for ig in in_grad))
				
				encoder_decoder.gnn_embedder.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig 
						  if ig is not None else None for ig in in_grad))
				
				encoder_decoder.decoders.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig
						  if ig is not None else None for ig in in_grad))

			'''
			encoder.register_full_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(config.encoder_grad_frac * ig 
												 if ig is not None else None for ig in in_grad))
			'''

		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, r2e=r2e, 
			contrastive_loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
			metrics=dict(Accuracy=self._contrastive_accuracy, Mask_pct=self._mask_pct), 
			save_model_directory=config.save_model_directory, **kwargs)
		
		self.mask_rate = config.mask_rate
		self.mask_span = config.mask_span
		self.temp = config.temp
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

	'''
	def _generate_negatives(self, z):
		"""Generate negative samples to compare each sequence location against"""
		batch_size, feat, full_len = z.shape
		z_k = z.permute([0, 2, 1]).reshape(-1, feat)
		with torch.no_grad():
			negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * self.num_negatives))
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

			for i in range(1, batch_size):
				negative_inds[i] += i * full_len

		z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, self.num_negatives, feat)
		return z_k, negative_inds

	def _calculate_similarity(self, z, c, negatives):
		c = c.permute([0, 2, 1]).unsqueeze(-2)
		z = z.permute([0, 2, 1]).unsqueeze(-2)

		# In case the contextualizer matches exactly, need to avoid divide by zero errors
		negative_in_target = (c == negatives).all(-1)
		targets = torch.cat([c, negatives], dim=-2)

		logits = F.cosine_similarity(z, targets, dim=-1) / self.temp

		if negative_in_target.any():
			# BENDR implementation is actually wrong...
			logits[..., 1:][negative_in_target] = float("-inf")

		return logits.view(-1, logits.shape[-1])
	'''

	def forward(self, data):
		relevant_bands = [data[band].float().to(self.device) for band in BANDS]
		inputs = dict(zip(BANDS, relevant_bands))
		encoder_output = self.encoder(data['graph'], inputs)
		contextualizer_output, shape, wavelet_embeddings = self.contextualizer(encoder_output)
		loss, correct, pairs = self.leave_one_out(wavelet_embeddings, self.contrastive_loss_fn)

		'''
		Convert z and negatives into SPD matrices 
		self.contextualizer.freeze_features(unfreeze=False)
		self.r2e.freeze_features(unfreeze=False)

		spd_z, shape = self.contextualizer(unmasked_z)
		spd_z = self.r2e(spd_z, shape)

		spd_negatives, shape = self.contextualizer(negatives[:, :, 0, :].permute([0, 2, 1]))
		spd_negatives = self.r2e(spd_negatives, shape)

		self.contextualizer.freeze_features(unfreeze=True)
		self.r2e.freeze_features(unfreeze=True)

		# Prediction -> batch_size x predict_length x predict_length
		logits = self._calculate_similarity(z=spd_z, c=c, negatives=spd_negatives.permute([0, 2, 1])[:, :, None, :])
		return logits, z, mask, decoded_coefficients
		'''
		return
	
	def calculate_loss(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
		recon_loss = self._reconstruction_loss(inputs, wavelet_recons)
		return self.loss_fn(logits, labels) + self.alpha * outputs[1].pow(2).mean() + recon_loss, recon_loss

	def _reconstruction_loss(self, input, outputs):
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
		modalities = list(embeddings.keys())
		num_targets = len(modalities)
		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			# Average embeddings of all other modalities
			other_emb = torch.stack(
				[embeddings[modalities[j]] for j in list(range(i)) + list(range(i + 1, num_targets))]
			).sum(0) / (num_targets - 1)

			# Compute logits
			logits = torch.matmul(embeddings[modalities[i]], other_emb.T) * torch.exp(self.temp)
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
	
	def sample_wise_contrastive(self, wavelet_spd_embeddings, criterion):
		"""
		Compute Sample Wise Contrastive Loss for wavelet spd embeddings. 

		Args:
			wavelet_spd_embeddings (dict): Dictionary of wavelet embeddings as SPD matrices.
			criterion: Loss function (e.g., CrossEntropyLoss).
			temperature (torch.nn.Parameter): Temperature parameter for scaling logits.

		Returns:
			loss (torch.Tensor): Total leave-one-out loss.
			correct (int): Number of correct predictions.
			pairs (int): Number of prediction pairs.
		"""

		modalities = list(embeddings.keys())
		num_targets = len(modalities)
		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			for j in range(i + 1, num_targets):
				emb_i = embeddings[modalities[i]]
				emb_j = embeddings[modalities[j]]

				# Compute logits
				logits = torch.matmul(emb_i, emb_j.T) * torch.exp(temperature)
				labels = torch.arange(logits.shape[0], device=device)

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





	@staticmethod
	def _mask_pct(inputs, outputs):
		return outputs[2].float().mean().item()

	@staticmethod
	def _simple_accuracy(inputs, outputs):
		if isinstance(outputs, (list, tuple)):
			outputs = outputs[0]
        # average over last dimensions
		while len(outputs.shape) >= 3:
			outputs = outputs.mean(dim=-1)
		return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()
	