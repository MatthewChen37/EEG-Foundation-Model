import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, SPDRectified, WaveletLogEuclideanMean, tensor_exp, tensor_log, AttentionManifold
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import PositionalEncoding, _make_mask_idxes
from ..ManifoldTransformer import ManifoldTransformer
from einops import rearrange

'''
BENDR-style Contextualizer using MAtt module 
'''
class MENDRContextualizerLarge(nn.Module):
	def __init__(self, 
				wavelet_contextualizer,
				combined_contextualizer):
		super().__init__()
		self.wavelet_contextualizer = wavelet_contextualizer	
		self.combined_contextualizer = combined_contextualizer

	def forward(self, x, batch_size, num_patches):
		wavelet_manifold_output = self.wavelet_contextualizer(x, batch_size, num_patches)
		# Never mask when calling it from here
		combined_manifold_output, mask_idxes = self.combined_contextualizer(wavelet_manifold_output, batch_size, num_patches, mask_ratio=0.0)
		return combined_manifold_output, wavelet_manifold_output, mask_idxes # Adding this for consistency of API

class MENDRWaveletContextualizer(nn.Module):
	def __init__(self, num_channels, out_dim,
				include_high=False,
				temp=10.0, patch_lens=None,
				n_transformer_layers=2):
		super().__init__()
		self.num_channels = num_channels
		self.out_dim = out_dim
		self.include_high = include_high

		if patch_lens is None:
			print("Note: Patch lengths are not specified, using default values. (2 second patches)")
			self.patch_lens = { # 2 second patches
				'delta': 8,
				'theta': 8,
				'alpha': 16,
				'beta': 32,
				'gamma': 64,
			}
			if self.include_high:
				self.patch_lens['high'] = 128
		else:
			self.patch_lens = patch_lens

		# Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(temp, requires_grad=True), requires_grad=True)

		# Positional Encoding
		self.position_encoder = dict()
		for band in self.patch_lens:
			self.position_encoder[band] = PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens[band])
		self.position_encoder = nn.ParameterDict(self.position_encoder)

		self.wavelet_mlp = dict()
		for band in self.patch_lens:
			self.wavelet_mlp[band] = nn.Sequential(nn.GELU(),
			nn.Linear(self.patch_lens[band]*out_dim, self.patch_lens[band]*out_dim),
			nn.GELU(),
			nn.Linear(self.patch_lens[band]*out_dim, self.patch_lens[band]*out_dim))
		self.wavelet_mlp = nn.ParameterDict(self.wavelet_mlp)

		self.wavelet_e2r = dict()
		for band in self.patch_lens:
			self.wavelet_e2r[band] = E2R()
		self.wavelet_e2r = nn.ParameterDict(self.wavelet_e2r)

		self.pre_attention_spd_transform = dict()
		for band in self.patch_lens:
			self.pre_attention_spd_transform[band] = SPDTransform(self.num_channels, self.num_channels)
		self.pre_attention_spd_transform = nn.ParameterDict(self.pre_attention_spd_transform)

		self.wavelet_manifold_transformers = dict()
		for band in self.patch_lens:
			self.wavelet_manifold_transformers[band] = []
			for i in range(n_transformer_layers):
				if i == n_transformer_layers - 1:
					self.wavelet_manifold_transformers[band].append(ManifoldTransformer(self.num_channels, norm_output=False, hidden_scale=1.5))
				else:
					self.wavelet_manifold_transformers[band].append(ManifoldTransformer(self.num_channels, hidden_scale=1.5))
			self.wavelet_manifold_transformers[band] = nn.ModuleList(self.wavelet_manifold_transformers[band])
		self.wavelet_manifold_transformers = nn.ParameterDict(self.wavelet_manifold_transformers)

		self.wavelet_tangent_space = dict()
		for band in self.patch_lens:
			self.wavelet_tangent_space[band] = SPDTangentSpace(self.num_channels)
		self.wavelet_tangent_space = nn.ParameterDict(self.wavelet_tangent_space)

	def forward(self, x, batch_size, num_patches):
		#assert x.keys() == self.wavelet_attention_manifolds.keys()
		# Batch Size, Num of Channels, Time Length
		cov_matrices = dict()
		for band in x.keys():
			x[band] = input_x + self.position_encoder[band](x[band])
			x[band] = self.wavelet_mlp[band](x[band])
			cov_matrices[band] = self.e2r(x[band])
		wavelet_manifold_output = dict()
		for band, band_encodings in cov_matrices.items():
			wavelet_manifold_output[band] = rearrange(band_encodings[band], 'B P C1 C2 -> (B P) C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
			wavelet_manifold_output[band] = self.pre_attention_spd_transform[band](wavelet_manifold_output[band])
			wavelet_manifold_output[band] = rearrange(wavelet_manifold_output[band], '(B P) C1 C2 -> B P C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
			for transformer in self.wavelet_manifold_transformers[band]:
				# output shape is [B, P, N, N]
				wavelet_manifold_output[band] = transformer(wavelet_manifold_output[band], batch_size, num_patches)
		return wavelet_manifold_output

class MENDRCombinedContextualizer(nn.Module):
	def __init__(self, num_channels, n_transformer_layers=4):
		super().__init__()
		self.num_channels = num_channels
		self.attention_manifold = AttentionManifold(self.num_channels, self.num_channels, heads=1)

		assert n_transformer_layers >= 1, "Must have at least one transformer layer"

		manifold_transformers = []
		for i in range(n_transformer_layers):
			if i == n_transformer_layers - 1:
				manifold_transformers.append(ManifoldTransformer(self.num_channels, norm_output=False, hidden_scale=1.5))
			else:
				manifold_transformers.append(ManifoldTransformer(self.num_channels, hidden_scale=1.5))

		self.manifold_transformer = nn.ModuleList(manifold_transformers)
		
		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.num_channels, self.num_channels)).float()
		self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, wavelet_manifold_output, batch_size, num_patches, mask_ratio=0.0):
		x = []
		for band in wavelet_manifold_output.keys():
			wavelet_manifold_output[band] = rearrange(wavelet_manifold_output[band], 'B P C1 C2 -> (B P) C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
			x.append(wavelet_manifold_output[band])
		x = torch.stack(x, dim=1)
		#print("Before Attention Manifold: ", x.shape)
		x, shape = self.attention_manifold(x)
		#print("After Attention Manifold: ", x.shape)
		x = rearrange(x, '(B P) C1 C2 -> B P C1 C2', B=shape[0], P=shape[1], C1=self.num_channels, C2=self.num_channels)
		transformed_dict = dict()
		for band_idx, band in enumerate(wavelet_manifold_output.keys()):
			transformed_dict[band] = rearrange(x[:, band_idx, :, :], '(B P) C1 C2 -> B P C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
		combined_manifold_output = WaveletLogEuclideanMean(transformed_dict)
		# Combined Manifold Output should be a clone
		combined_manifold_output_hidden_dim = combined_manifold_output.shape[-1]
		mask_idxes = None
		if mask_ratio > 0.0:
			x = combined_manifold_output.clone() # Just in case
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			combined_manifold_output = rearrange(combined_manifold_output, 'B P C1 C2 -> (B P) C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
			# We randomly mask each patch with probability mask_ratio
			# and calculate the LEM and then compare it with the full LEM
			# [B, P, C, C]
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			combined_manifold_output[mask_idxes] = spd_mask

			'''
			print(mask_idxes.shape)
			for batch_idx in range(combined_manifold_output.shape[0]):
				for patch_idx in range(combined_manifold_output.shape[1]):
					if mask_idxes[batch_idx, patch_idx] == False:
						print("HERE", mask_idxes.shape)
						assert not torch.equal(combined_manifold_output[batch_idx, patch_idx], torch.zeros(combined_manifold_output[batch_idx, patch_idx].shape, device=self.device)), f"Masked SPD Matrix {batch_idx, patch_idx} is not equal to the mask"
			'''
		else:
			x = combined_manifold_output.clone()
		for transformer in self.manifold_transformer:
			x = transformer(x, batch_size, num_patches)
		return x, mask_idxes