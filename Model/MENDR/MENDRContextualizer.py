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

		self.transformer_layers = 2	
		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'theta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'alpha': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'beta': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'gamma': nn.Sequential(
				E2R(epochs=4, device=self.device),
				AttentionManifold(19, 32, self.device)
			)
		})

		self.wavelet_r2e_tangent_spaces = nn.ParameterDict({
			'delta': SPDTangentSpace(32),
			'theta': SPDTangentSpace(32),
			'alpha': SPDTangentSpace(32),
			'beta': SPDTangentSpace(32),
			'gamma': SPDTangentSpace(32),
		})

		self.wavelet_r2e_lin = nn.ParameterDict({
			'delta': nn.Sequential({
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(80, 80),
			}),
			'theta': nn.Sequential({
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(80, 80),
			}),
			'alpha': nn.Sequential({
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(80, 80),
			}),
			'beta': nn.Sequential({
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(80, 80),
			}),
			'gamma': nn.Sequential({
				nn.Flatten(),
				nn.GELU(),
				nn.Linear(80, 80),
			})
		})

		self.combined_r2e_tangent_space = nn.Sequential(
			SPDTangentSpace(32)
		)

		self.combined_r2e_lin = nn.Sequential(
			nn.Flatten(),
			nn.GELU(),
			nn.Linear(80, 80),
		)

		self.combined_attention = AttentionManifold(32, 32, self.device)
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
			wavelet_r2e_output[band] = self.wavelet_r2e_output[band].view(shape[0], shape[1], -1)
			wavelet_r2e_output[band] = self.wavelet_r2e_lin[band](wavelet_r2e_output[band])

		# The sum of SPD matrices is also SPD
		combined_wavelet_spd = torch.zeros(embedding_shapes['delta'][0], 4, 32, 32).to(self.device)
		for band in wavelet_output.keys():
			output = wavelet_output[band][0]
			shape  = wavelet_output[band][1]
			output = output.view((shape[0], shape[1], 32, 32))
			combined_wavelet_spd += output

		combined_manifold_output, shape = self.combined_attention(combined_wavelet_spd)
		combined_r2e_output = self.ract(combined_manifold_output)
		combined_r2e_output = self.combined_r2e_tangent_space(combined_r2e_output)
		combined_r2e_output = combined_r2e_output.shape(shape[0], shape[1], -1)
		combined_r2e_output = self.combined_r2e_lin(combined_manifold_output)
		return combined_r2e_output, combined_manifold_output, wavelet_r2e_output, wavelet_manifold_output

	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False