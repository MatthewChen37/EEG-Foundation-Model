import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from ..layers import Permute, Flatten
import math

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizer(nn.Module):
	def __init__(self, device, epochs=4, num_channels=19):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		self.epochs = epochs
		self.channels = num_channels

		self.WaveletContextualizer = MENDRWaveletContextualizer(device=self.device, epochs=self.epochs, num_channels=self.channels)
		self.CombinedContextualizer = MENDRCombinedContextualizer(device=self.device, num_channels=self.channels)

	def forward(self, x):
		wavelet_manifold_output, epoched_shape = self.WaveletContextualizer(x)
		combined_r2e_output, combined_manifold_output = self.CombinedContextualizer(wavelet_manifold_output, epoched_shape)
		return combined_r2e_output, combined_manifold_output, wavelet_manifold_output

	def freeze_features(self, unfreeze=False):
		for param in self.parameters():
			param.requires_grad = unfreeze

class MENDRWaveletContextualizer(nn.Module):
	def __init__(self, device, epochs=4, num_channels=19):
		super(MENDRWaveletContextualizer, self).__init__()
		self.device = device
		self.epochs = epochs

		self.position_encoder = nn.ParameterDict({
			'delta': PositionalEncoding(num_channels, 124, 0, 0.1, self.epochs),
			'theta': PositionalEncoding(num_channels, 124, 0, 0.1, self.epochs),
			'alpha': PositionalEncoding(num_channels, 244, 0, 0.1, self.epochs),
			'beta': PositionalEncoding(num_channels, 484, 0, 0.1, self.epochs),
			'gamma': PositionalEncoding(num_channels, 484, 0, 0.1, self.epochs),
		}).to(self.device)

		self.wavelet_e2r = nn.ParameterDict({
			'delta': E2R(epochs=self.epochs, device=self.device),
			'theta': E2R(epochs=self.epochs, device=self.device),
			'alpha': E2R(epochs=self.epochs, device=self.device),
			'beta': E2R(epochs=self.epochs, device=self.device),
			'gamma': E2R(epochs=self.epochs, device=self.device),
		})

		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': AttentionManifold(num_channels, num_channels, self.device),
			'theta': AttentionManifold(num_channels, num_channels, self.device),
			'alpha': AttentionManifold(num_channels, num_channels, self.device),
			'beta': AttentionManifold(num_channels, num_channels, self.device),
			'gamma': AttentionManifold(num_channels, num_channels, self.device),
		})

		self.trace_normalization = BatchTraceNormalization(self.device)

		self.wavelet_spd_transforms = nn.ParameterDict({
			'delta': SPDTransform(num_channels, num_channels, self.device),
			'theta': SPDTransform(num_channels, num_channels, self.device),
			'alpha': SPDTransform(num_channels, num_channels, self.device),
			'beta': SPDTransform(num_channels, num_channels, self.device),
			'gamma': SPDTransform(num_channels, num_channels, self.device)
		})

		self.ract = SPDRectified()

	def forward(self, x):
		#assert x.keys() == self.wavelet_attention_manifolds.keys()
		# Batch Size, Num of Channels, Time Length
		batch_size = x['delta'][0].shape[0]
		x_input = dict()
		for band in x.keys():
			if self.position_encoder:
				x_input[band] = self.position_encoder[band](x[band][0])
			else:
				x_input[band] = x[band][0]

		epoched_shape = None
		wavelet_manifold_output = dict()
		for band, band_encodings in x_input.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings)
			#assert torch.allclose(wavelet_manifold_output[band], wavelet_manifold_output[band].mT, atol=(10 ** -10))
			output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band])
			#assert torch.allclose(output, output.mT, atol=(10 ** -10)), "Attention Manifold"
			# Skip Connection
			epoched_shape = shape
			og_output_shape = output.shape
			output = output.view(wavelet_manifold_output[band].shape) + wavelet_manifold_output[band]
			output = output.view(og_output_shape) 
			#assert torch.allclose(output, output.mT, atol=(10 ** -10))
			output = self.trace_normalization(output) # If I comment this out, then matrix is not SPD anymore???
			wavelet_manifold_output[band] = self.ract(output)
			wavelet_manifold_output[band] = self.wavelet_spd_transforms[band](output)

		return wavelet_manifold_output, epoched_shape

class MENDRCombinedContextualizer(nn.Module):
	def __init__(self, device, num_channels=19):
		super().__init__()
		self.device = device
		self.num_channels = num_channels

		self.combined_attention = AttentionManifold(self.num_channels, self.num_channels, self.device)
		self.combined_spd_transform1 = nn.Sequential(SPDTransform(self.num_channels, self.num_channels, self.device), BatchTraceNormalization(self.device))
		self.combined_r2e_tangent_space = SPDTangentSpace(self.num_channels, self.device)
		self.combined_spd_transform2 = SPDTransform(self.num_channels, self.num_channels, self.device)
		self.ract = SPDRectified()
		self.flatten = nn.Flatten()

	def forward(self, x, og_output_shape):
		# Log Euclidean Mean
		combined_manifold_output = torch.stack(list(x.values()), dim=1)
		combined_manifold_output = self.combined_attention.tensor_log(combined_manifold_output)
		combined_manifold_output = self.combined_attention.tensor_exp((combined_manifold_output.sum(dim=1, keepdim=True)) / combined_manifold_output.shape[1])[:, 0, :, :]

		combined_manifold_output, shape = self.combined_attention(combined_manifold_output, shape=og_output_shape)
		combined_manifold_output = self.ract(combined_manifold_output)

		combined_manifold_output = self.combined_spd_transform1(combined_manifold_output)
		combined_manifold_output = self.ract(combined_manifold_output)
		combined_manifold_output = self.combined_spd_transform2(combined_manifold_output)
		combined_r2e_output = self.combined_r2e_tangent_space(combined_manifold_output)
		combined_r2e_output = combined_r2e_output.view(shape[0], shape[1], -1)
		combined_r2e_output = self.flatten(combined_r2e_output)
		return combined_r2e_output, combined_manifold_output


class BatchTraceNormalization(nn.Module):
	def __init__(self, device, num_channels=19, epsilon=1E-5):
		super().__init__()
		self.num_channels = num_channels
		self.epsilon = epsilon
		self.device = device

	def forward(self, x):
		# Expects [B, C, C]
		trace = x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
		trace = trace.view(-1, 1, 1)
		trace = trace + self.epsilon*torch.ones(trace.shape).to(self.device)
		x /= trace
		identity = torch.eye(x.shape[-1], x.shape[-1], device=self.device).to(self.device).repeat(x.shape[0], 1, 1)
		x = x + (1e5 * identity)
		return x


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