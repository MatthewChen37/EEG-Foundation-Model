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
	def __init__(self, device):
		super(MENDRContextualizer, self).__init__()
		self.device = device

		self.wavelet_e2r = nn.ParameterDict({
			'delta': E2R(epochs=4, device=self.device),
			'theta': E2R(epochs=4, device=self.device),
			'alpha': E2R(epochs=4, device=self.device),
			'beta': E2R(epochs=4, device=self.device),
			'gamma': E2R(epochs=4, device=self.device),
		})

		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': AttentionManifold(19, 19, self.device),
			'theta': AttentionManifold(19, 19, self.device),
			'alpha': AttentionManifold(19, 19, self.device),
			'beta': AttentionManifold(19, 19, self.device),
			'gamma': AttentionManifold(19, 19, self.device)
		})

		self.wavelet_r2e_tangent_spaces = nn.ParameterDict({
			'delta': SPDTangentSpace(19, self.device),
			'theta': SPDTangentSpace(19, self.device),
			'alpha': SPDTangentSpace(19, self.device),
			'beta': SPDTangentSpace(19, self.device),
			'gamma': SPDTangentSpace(19, self.device),
		})

		self.wavelet_r2e_lin = nn.ParameterDict({
			'delta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Dropout(p=0.1),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 190),
				nn.GELU(),
				nn.Linear(190, 95),
			),
			'theta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Dropout(p=0.1),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 190),
				nn.GELU(),
				nn.Linear(190, 95),
			),
			'alpha': nn.Sequential(
				nn.Flatten(),
				nn.Dropout(p=0.1),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 190),
				nn.GELU(),
				nn.Linear(190, 95),
			),
			'beta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 190),
				nn.GELU(),
				nn.Linear(190, 95),
			),
			'gamma': nn.Sequential(
				nn.Flatten(),
				nn.Dropout(p=0.1),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 190),
				nn.GELU(),
				nn.Linear(190, 95),
			)
		}).to(self.device)

		self.combined_r2e_tangent_space = nn.Sequential(
			SPDTangentSpace(19, self.device)
		)

		self.combined_r2e_lin = nn.Sequential(
			nn.Flatten(),
			nn.GELU(),
			nn.Dropout(p=0.1),
			nn.GroupNorm(1, 760),
			nn.Linear(760, 190),
			nn.GELU(),
			nn.Linear(190, 95)
		).to(self.device)

		self.combined_attention = AttentionManifold(19, 19, self.device)
		self.ract = SPDRectified()

		self.apply(self.init_params)

		self.position_encoder = nn.ParameterDict({
			'delta': PositionalEncoding(19, 124, 0, 0.1, 4),
			'theta': PositionalEncoding(19, 124, 0, 0.1, 4),
			'alpha': PositionalEncoding(19, 244, 0, 0.1, 4),
			'beta': PositionalEncoding(19, 484, 0, 0.1, 4),
			'gamma': PositionalEncoding(19, 642, 2, 0.1, 4),
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
		wavelet_r2e_output = dict()
		for band, band_encodings_decodings in x.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings_decodings[0])
			output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band])
			# Skip Connection
			og_output_shape = output.shape
			output = output.view(wavelet_manifold_output[band].shape) + wavelet_manifold_output[band]
			output = output.view(og_output_shape) 
			wavelet_manifold_output[band] = (self.ract(output), shape)
			wavelet_r2e_output[band] = self.wavelet_r2e_tangent_spaces[band](output)
			wavelet_r2e_output[band] = wavelet_r2e_output[band].view(shape[0], shape[1], -1)
			wavelet_r2e_output[band] = self.wavelet_r2e_lin[band](wavelet_r2e_output[band])

		# The sum of SPD matrices is also SPD
		combined_manifold_output = torch.zeros(embedding_shapes['delta'][0], 4, 19, 19).to(self.device)
		for band in wavelet_manifold_output.keys():
			output = wavelet_manifold_output[band][0]
			shape  = wavelet_manifold_output[band][1]
			output = output.view((shape[0], shape[1], 19, 19))
			combined_manifold_output += output

		combined_manifold_output_skip, shape = self.combined_attention(combined_manifold_output)
		# Skip connection
		combined_manifold_output =  combined_manifold_output.view(shape[0]*shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3]) + combined_manifold_output_skip
		combined_r2e_output = self.combined_r2e_tangent_space(combined_manifold_output)
		combined_r2e_output = combined_r2e_output.view(shape[0], shape[1], -1)
		combined_r2e_output = self.combined_r2e_lin(combined_r2e_output)
		return combined_r2e_output, combined_manifold_output, wavelet_r2e_output, wavelet_manifold_output

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

		conv = nn.Conv1d(self.len, self.len, self.channels, padding= self.channels // 2, groups=self.epochs)
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