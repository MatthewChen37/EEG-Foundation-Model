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
			encoder.register_full_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(config.encoder_grad_frac * ig 
												 if ig is not None else None for ig in in_grad))
			
		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, r2e=r2e, 
			loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
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

			
	def forward(self, data):
		relevant_bands = [data[band].float().to(self.device) for band in BANDS]
		inputs = dict(zip(BANDS, relevant_bands))
		encoder_output = self.encoder(data['graph'], inputs)
		contextualizer_output = self.contextualizer(encoder_output)
		'''
		unmasked_z = z.clone()
		batch_size, feat, samples = z.shape

		if self._training:
			mask = _make_mask((batch_size, samples), self.mask_rate, samples, self.mask_span)
		else:
			mask = torch.zeros((batch_size, samples), requires_grad=False, dtype=torch.bool)
			half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
			if samples <= self.mask_span * half_avg_num_seeds:
				raise ValueError("Masking the entire span, pointless.")
			mask[:, _make_span_from_seeds((samples // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int),
												self.mask_span)] = True

		c, shape = self.contextualizer(z, mask_t=mask)
		c = self.r2e(c, shape)

		# Select negative candidates and generate labels for which are correct labels
		negatives, negative_inds = self._generate_negatives(z)

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
	