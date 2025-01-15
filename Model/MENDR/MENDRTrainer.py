import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# TODO: Bad Practice fix later
import sys
sys.path.append("../")
from ..baseModelTrainer import BaseModelTrainer
from ..contextualizer import _make_mask, _make_span_from_seeds
from ..TrainingDecoder.trainingDecoder import ConvDecoder
from torch_geometric.utils import unbatch

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, stembedder, encoder, contextualizer, r2e, decoder, config, **kwargs):
		'''
		Args:
			stembedder: a nn.Module
			encoder: a nn.Module
			contextualizer: a nn.Module
			r2e: a nn.Module
			config: a dictionary containing the following keys:
				- mask_span: an integer
				- multi_gpu: a boolean
				- encoder_grad_frac: a float
				- learning_rate: a float
				- l2_weight_decay: a float	
				- mask_rate: a float
				- temp: a float
				- permuted_contexts: a boolean
				- enc_feat_l2: a float
				- num_negatives: an integer
		'''
		self._enc_downsample = encoder.downsampling_factor
		if config.multi_gpu:
			stembedder = nn.DataParallel(stembedder)
			encoder = nn.DataParallel(encoder)
			contextualizer = nn.DataParallel(contextualizer)
			r2e = nn.DataParallel(r2e)
		if config.encoder_grad_frac < 1:
            # TODO: I hope this works...
			encoder.register_full_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(config.encoder_grad_frac * ig 
												 if ig is not None else None for ig in in_grad))
			
		super(MENDRTrainer, self).__init__(embedder=stembedder, encoder=encoder, contextualizer=contextualizer, r2e=r2e, decoder=decoder,
			loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
			metrics=dict(Accuracy=self._contrastive_accuracy, Mask_pct=self._mask_pct), **kwargs)
		
		self.mask_rate = config.mask_rate
		self.mask_span = config.mask_span
		self.temp = config.temp
		self.permuted_contexts = config.permuted_contexts
		self.alpha = config.enc_feat_l2
		self.beta = 0.5
		self.start_token = getattr(contextualizer, 'start_token', None)
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

			
	def forward(self, *inputs):
		z = self.embedder(inputs[0])
		z = self.encoder(z)

		decoded_signal = self.decoder(z)

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

		'''
		Convert z and negatives into SPD matrices 
		'''
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
		return logits, z, mask, decoded_signal
	
	def calculate_loss(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
		return self.loss_fn(logits, labels) + self.alpha * outputs[1].pow(2).mean() + self.beta * self._reconstruction_loss(inputs, outputs[3])
	
	def _reconstruction_loss(self, original, reconstruction):
		# TODO: We really don't need this if statment...
		# Mean Squared Error
		if isinstance(original, list):
			signals_from_graphs = torch.stack(original).float().to(self.device)
		else:
			signals_from_graphs = torch.tensor(np.vstack(original.x)).float().to(self.device)
			signals_from_graphs = unbatch(signals_from_graphs, original.batch)
			signals_from_graphs = torch.stack(signals_from_graphs)
		assert signals_from_graphs.shape == reconstruction.shape
		return F.mse_loss(signals_from_graphs, reconstruction)
    
	def _contrastive_accuracy(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		return self._simple_accuracy([labels], logits)
    
	def calculate_metrics(self, *inputs, outputs):
		"""
		Cosine Similarity from Calculating Similarity
		"""
		# "Logits" from _calculate_similarity
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)

		return {
			'Contrastive Accuracy': self._simple_accuracy([labels], logits),
			'MASK_pct': self._mask_pct(inputs, outputs),
			'BENDR Reconstruction MSE': self._reconstruction_loss(inputs[0], outputs[3]).item()
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
	
if __name__ == "__main__":
	from encoder import ConvEncoder
	from MENDRContextualizer import mATTContextualizer
	from R2E import R2E
	from types import SimpleNamespace
	from contextualizer import Contextualizer

	encoder = ConvEncoder(in_features=19, encoder_h=256, enc_width=(3, 2, 2),
                          dropout=0., enc_downsample=(3, 2, 2))


	contextualizer_config = SimpleNamespace(
		in_features=256,
		dropout=0.1,
		start_token=-5,
		position_encoder=25,
		epochs=4
	)

	contextualizer = mATTContextualizer(contextualizer_config)
	r2e = R2E(contextualizer_config)


	trainer_config = SimpleNamespace(
		mask_span=6,
		multi_gpu=False,
		encoder_grad_frac=1,
		mask_rate=0.1,
		temp=0.1,
		permuted_encodings=False,
		permuted_contexts=False,
		enc_feat_l2=1e-5,
		num_negatives=10,
		learning_rate=1e-3,
		l2_weight_decay=1e-5,

	)

	decoder = ConvDecoder(encoder_h=256, out_features=19, dec_width=(2, 2, 3),
						  dropout=0., dec_upsample=(2, 2, 3), original_time_len=100,
						  top_k=3, num_kernels=3)

	trainer = MENDRTrainer(encoder, contextualizer, r2e, decoder, trainer_config)
	print(trainer.description(27000))

	input = torch.ones(3, 19, 1000)
	output = trainer.forward(input)
	print("Input shape:", input.shape, "Output shape:", output[0].shape)