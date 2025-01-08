import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baseModelTrainer import BaseModelTrainer
from contextualizer import _make_mask, _make_span_from_seeds

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, encoder, contextualizer, r2e, config, **kwargs):
		'''
		Args:
			encoder: a nn.Module
			contextualizer: a nn.Module
			r2e: a nn.Module
			config: a dictionary containing the following keys:
				- mask_span: an integer
				- multi_gpu: a boolean
				- encoder_grad_frac: a float
				- learning_rate: a float
				- l2_weight_decay: a float	
		'''
		self.predict_length = config.mask_span
		self._enc_downsample = encoder.downsampling_factor
		if config.multi_gpu:
			encoder = nn.DataParallel(encoder)
			context_fn = nn.DataParallel(context_fn)
		if config.encoder_grad_frac < 1:
            # TODO: I hope this works...
			encoder.register_full_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(config.encoder_grad_frac * ig if ig is not None else None for ig in in_grad))
			
		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, r2e=r2e,
			loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
			metrics=dict(Accuracy=self._contrastive_accuracy, Mask_pct=self._mask_pct), **kwargs)
		
		self.best_metric = None
		self.mask_rate = config.mask_rate
		self.mask_span = config.mask_span
		self.temp = config.temp
		self.permuted_encodings = config.permuted_encodings
		self.permuted_contexts = config.permuted_contexts
		self.beta = config.enc_feat_l2
		self.start_token = getattr(context_fn, 'start_token', None)
		self.unmasked_negative_frac = config.unmasked_negative_frac
		self.num_negatives = config.num_negatives

	def description(self, sequence_len):
		encoded_samples = self._enc_downsample(sequence_len)
		desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
			encoded_samples, self.mask_span, self.mask_rate,
			int(encoded_samples * self.mask_rate * self.mask_span))
		return desc
	
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
		c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
		z = z.permute([0, 2, 1]).unsqueeze(-2)

		# In case the contextualizer matches exactly, need to avoid divide by zero errors
		negative_in_target = (c == negatives).all(-1)
		targets = torch.cat([c, negatives], dim=-2)

		logits = F.cosine_similarity(z, targets, dim=-1) / self.temp
		if negative_in_target.any():
			logits[1:][negative_in_target] = float("-inf")

		return logits.view(-1, logits.shape[-1])

			
	def forward(self, *inputs):
		z = self.encoder(inputs[0])

		if self.permuted_encodings:
			z = z.permute([1, 2, 0])

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

		c = self.context_fn(z, mask)

		# Select negative candidates and generate labels for which are correct labels
		negatives, negative_inds = self._generate_negatives(z)

		# Prediction -> batch_size x predict_length x predict_length
		logits = self._calculate_similarity(z=unmasked_z, c=c, negatives=negatives)
		return logits, z, mask
	
	def calculate_loss(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
		return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()
    
	def _contrastive_accuracy(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		return self._simple_accuracy([labels], logits)
    
	def calculate_metrics(self, inputs, outputs):
		"""
		Cosine Similarity from Calculating Similarity
		"""
		# "Logits" from _calculate_similarity
		similarity = outputs[0].mean().item()
		return {
			'Similarity': similarity
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