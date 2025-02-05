import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace
from ..layers import Permute, Flatten

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizer(nn.Module):
	def __init__(self, device):
		super(MENDRContextualizer, self).__init__()
		self.device = device

		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 10, self.device)
			),
			'theta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 10, self.device)
			),
			'alpha': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 10, self.device)
			),
			'beta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 10, self.device)
			),
			'gamma': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 10, self.device)
			)
		})

		self.wavelet_r2e_tangent_spaces = nn.ParameterDict({
			'delta': SPDTangentSpace(10),
			'theta': SPDTangentSpace(10),
			'alpha': SPDTangentSpace(10),
			'beta': SPDTangentSpace(10),
			'gamma': SPDTangentSpace(10),
		})

		self.wavelet_r2e_lin = nn.ParameterDict({
			'delta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(220, 110),
			),
			'theta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(220, 110),
			),
			'alpha': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(220, 110),
			),
			'beta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(220, 110),
			),
			'gamma': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(220, 110),
			)
		})

		self.combined_r2e_tangent_space = nn.Sequential(
			SPDTangentSpace(10)
		)

		self.combined_r2e_lin = nn.Sequential(
			nn.Flatten(),
			nn.GELU(),
			nn.Linear(64, 64)
		)

		self.combined_attention = AttentionManifold(10, 8, self.device)
		self.ract = SPDRectified()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def forward(self, x, mask_t=None, mask_c=None):
		assert x.keys() == self.wavelet_attention_manifolds.keys()
		embedding_shapes = dict()
		for band in x.keys():
			# Batch Size, Num of Channels, Time Length
			embedding_shapes[band] = x[band][0].shape

		wavelet_manifold_output = dict()
		wavelet_r2e_output = dict()
		for band, band_encodings_decodings in x.items():
			wavelet_manifold_output[band] = self.wavelet_attention_manifolds[band](band_encodings_decodings[0])
			output = wavelet_manifold_output[band][0]
			shape  = wavelet_manifold_output[band][1]
			wavelet_manifold_output[band] = (self.ract(output), shape)
			wavelet_r2e_output[band] = self.wavelet_r2e_tangent_spaces[band](output)
			wavelet_r2e_output[band] = wavelet_r2e_output[band].view(shape[0], shape[1], -1)
			wavelet_r2e_output[band] = self.wavelet_r2e_lin[band](wavelet_r2e_output[band])

		# The sum of SPD matrices is also SPD
		combined_wavelet_spd = torch.zeros(embedding_shapes['delta'][0], 4, 10, 10).to(self.device)
		for band in wavelet_manifold_output.keys():
			output = wavelet_manifold_output[band][0]
			shape  = wavelet_manifold_output[band][1]
			output = output.view((shape[0], shape[1], 10, 10))
			combined_wavelet_spd += output

		combined_manifold_output, shape = self.combined_attention(combined_wavelet_spd)
		combined_r2e_output = self.ract(combined_manifold_output)
		combined_r2e_output = self.combined_r2e_tangent_space(combined_r2e_output)
		combined_r2e_output = combined_r2e_output.view(shape[0], shape[1], -1)
		combined_r2e_output = self.combined_r2e_lin(combined_manifold_output)
		return combined_r2e_output, combined_manifold_output, wavelet_r2e_output, wavelet_manifold_output

	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False