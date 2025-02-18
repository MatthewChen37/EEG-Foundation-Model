import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace
from ..layers import Permute, Flatten
import math

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizer(nn.Module):
	def __init__(self, device, epochs=4):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		self.epochs = epochs

		self.wavelet_e2r = nn.ParameterDict({
			'delta': E2R(epochs=self.epochs, device=self.device),
			'theta': E2R(epochs=self.epochs, device=self.device),
			'alpha': E2R(epochs=self.epochs, device=self.device),
			'beta': E2R(epochs=self.epochs, device=self.device),
			'gamma': E2R(epochs=self.epochs, device=self.device),
		})

		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': AttentionManifold(19, 19, self.device),
			'theta': AttentionManifold(19, 19, self.device),
			'alpha': AttentionManifold(19, 19, self.device),
			'beta': AttentionManifold(19, 19, self.device),
			'gamma': AttentionManifold(19, 19, self.device)
		})

		self.combined_r2e_tangent_space = SPDTangentSpace(19, self.device)
		self.combined_r2e_lin = nn.Sequential(
			nn.Flatten(),
			nn.BatchNorm1d(760),
			nn.Dropout(p=0.4),
			nn.Linear(760, 380),
			nn.GELU(),
			nn.BatchNorm1d(380),
			nn.Dropout(p=0.2),
			nn.Linear(380, 190)
		).to(self.device)

		self.combined_attention = AttentionManifold(19, 19, self.device)
		self.ract = SPDRectified()
		self.apply(self.init_params)
		self.position_encoder = nn.ParameterDict({
			'delta': PositionalEncoding(19, 124, 0, 0.1, self.epochs),
			'theta': PositionalEncoding(19, 124, 0, 0.1, self.epochs),
			'alpha': PositionalEncoding(19, 244, 0, 0.1, self.epochs),
			'beta': PositionalEncoding(19, 484, 0, 0.1, self.epochs),
			'gamma': PositionalEncoding(19, 484, 0, 0.1, self.epochs),
		}).to(self.device)

	def forward(self, x):
		assert x.keys() == self.wavelet_attention_manifolds.keys()
		embedding_shapes = dict()
		for band in x.keys():
			# Batch Size, Num of Channels, Time Length
			embedding_shapes[band] = x[band][0].shape
			if self.position_encoder:
				x[band] = (self.position_encoder[band](x[band][0]), x[band][1])

		wavelet_manifold_output = dict()
		for band, band_encodings_decodings in x.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings_decodings[0])
			output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band])
			# Skip Connection
			og_output_shape = output.shape
			output = output.view(wavelet_manifold_output[band].shape) + wavelet_manifold_output[band]
			output = output.view(og_output_shape) 
			wavelet_manifold_output[band] = (output, shape)

		# The sum of SPD matrices is also SPD
		combined_manifold_output = torch.zeros(embedding_shapes['delta'][0], 4, 19, 19).to(self.device)
		for band in wavelet_manifold_output.keys():
			output = wavelet_manifold_output[band][0]
			shape  = wavelet_manifold_output[band][1]
			output = output.view((shape[0], shape[1], 19, 19))
			combined_manifold_output += output

		combined_manifold_output, shape = self.combined_attention(combined_manifold_output)
		# Skip connection
		#combined_manifold_output =  combined_manifold_output.view(shape[0]*shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3]) + combined_manifold_output_skip
		combined_manifold_output = self.ract(combined_manifold_output)
		combined_r2e_output = self.combined_r2e_tangent_space(combined_manifold_output)
		combined_r2e_output = combined_r2e_output.view(shape[0], shape[1], -1)
		combined_r2e_output = self.combined_r2e_lin(combined_r2e_output)
		return combined_r2e_output, combined_manifold_output, wavelet_manifold_output

	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False

	def init_params(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight.data)
			if module.bias is not None:
				module.bias.data.zero_()

# Based on BENDR's Convolutional Position Encoding Scheme
class PositionalEncoding(nn.Module):
	def __init__(self, channels, seq_len, zero_padding=0, dropout=0.1, epochs=4):
		super().__init__()
		self.channels = channels
		self.epochs = epochs
		self.zero_padding = zero_padding

		if seq_len % self.epochs != 0:
			assert self.zero_padding != 0, f"Since seq len not divisible by epochs must add padding to make seq len divisible by epochs"
			assert (seq_len + self.zero_padding) % self.epochs == 0, f"Zero padding does not make seq divisble by epochs"
			self.len = seq_len + self.zero_padding
		else:
			self.len = seq_len

		conv = nn.Conv1d(self.len, self.len, self.channels, padding=self.channels // 2, groups=self.epochs)
		nn.init.normal_(conv.weight, mean=0, std=1)
		nn.init.constant_(conv.bias, 0)
		conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
		self.conv = nn.Sequential(conv, nn.GELU(), nn.Dropout(p=dropout))

	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape ``[batch_size, channels, seq_len]``
		"""
		B, C, L = x.shape
		if self.zero_padding:
			x = torch.cat([x, torch.zeros(B, C, self.zero_padding).to(x.device)], dim=-1)
		x = x.permute(0, 2, 1)
		positional_encoding = self.conv(x)
		x = x + positional_encoding
		x = x.permute(0, 2, 1)
		return x