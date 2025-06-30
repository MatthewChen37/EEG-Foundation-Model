import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified, WaveletLogEuclideanMean
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import (PositionalEncoding,
							_make_mask_idxes,
							BatchTraceNormalization,
							LogEuclidLayerNorm)
from Model.MENDR.Contextualizer.ManifoldTransformer import ManifoldTransformer, _RiemannianResidual
from einops import rearrange

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, num_channels, out_dim, include_high=False, patch_lens=None, contextualizer_layers=6):
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
			'''
			self.patch_lens = { # 5 second patches
				'delta': 20,
				'theta': 20,
				'alpha': 40,
				'beta': 80,
				'gamma': 160,
			}
			self.patch_lens = { # 10 second patches
				'delta': 40,
				'theta': 40,
				'alpha': 80,
				'beta': 160,
				'gamma': 320,
			}
			'''
			if self.include_high:
				self.patch_lens['high'] = 128
				#self.patch_lens['high'] = 320
				#self.patch_lens['high'] = 640
		else:
			self.patch_lens = patch_lens

		self.e2r = E2R()
		self.ract = SPDRectified()
		self.Contextualizer = MENDRContextualizer(
											encoded_out=self.num_channels,
											n_transformer_layers=contextualizer_layers)


		self.pre_attention_transform = SPDTransform(self.num_channels, self.num_channels)

		self.wavelet_mlp = {
			'delta': nn.Sequential(nn.GELU(), nn.Linear(self.patch_lens['delta']*out_dim, self.patch_lens['delta']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['delta']*out_dim, self.patch_lens['delta']*out_dim)),
			'theta': nn.Sequential(nn.GELU(), nn.Linear(self.patch_lens['theta']*out_dim, self.patch_lens['theta']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['theta']*out_dim, self.patch_lens['theta']*out_dim)),
			'alpha': nn.Sequential(nn.GELU(), nn.Linear(self.patch_lens['alpha']*out_dim, self.patch_lens['alpha']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['alpha']*out_dim, self.patch_lens['alpha']*out_dim)),
			'beta': nn.Sequential(nn.GELU(), nn.Linear(self.patch_lens['beta']*out_dim, self.patch_lens['beta']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['beta']*out_dim, self.patch_lens['beta']*out_dim)),
			'gamma': nn.Sequential(nn.GELU(), nn.Linear(self.patch_lens['gamma']*out_dim, self.patch_lens['gamma']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['gamma']*out_dim, self.patch_lens['gamma']*out_dim)),
		}

		self.position_encoders = {
			'delta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['delta']),
			'theta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['theta']),
			'alpha': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['alpha']),
			'beta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['beta']),
			'gamma': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['gamma']),
		}

		if self.include_high:
			self.position_encoders['high'] = PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['high'])
			self.wavelet_mlp['high'] = nn.Sequential(nn.Linear(self.patch_lens['high']*out_dim, self.patch_lens['high']*out_dim), nn.GELU(), nn.Linear(self.patch_lens['high']*out_dim, self.patch_lens['high']*out_dim))
		self.position_encoders = nn.ParameterDict(self.position_encoders)
		self.wavelet_mlp = nn.ParameterDict(self.wavelet_mlp)

	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is a dict of wavelet bands of shape [Batch, #patch, #num_channels, #time_step* #out_dim]
		# Each patch gets coagulated into the same time scale through the covariance matrix
		# Thus different patches will have different number of samples for calculating the covariance matrix
		cov_matrices = dict()
		for band in x.keys():
			x[band] = x[band] + self.position_encoders[band](x[band])
			x[band] = self.wavelet_mlp[band](x[band])
			cov_matrices[band] = self.e2r(x[band])
		combined_manifold_output = WaveletLogEuclideanMean(cov_matrices)
		# this is an issue: https://github.com/arogozhnikov/einops/issues/204
		combined_manifold_output = rearrange(combined_manifold_output, 'B P C1 C2 -> (B P) C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
		# This is a transformation to generate a "token embedding" for each patch
		combined_manifold_output = self.pre_attention_transform(combined_manifold_output)
		combined_manifold_output = rearrange(combined_manifold_output, '(B P) C1 C2 -> B P C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
		combined_manifold_output_masked, mask_idxes = self.Contextualizer(combined_manifold_output.clone(), batch_size, num_patches, mask_ratio)
		return combined_manifold_output, combined_manifold_output_masked, mask_idxes 
	
class MENDRContextualizer(nn.Module):
	def __init__(self, encoded_out, n_transformer_layers):
		super().__init__()
		self.encoded_out = encoded_out

		assert n_transformer_layers >= 1, "Must have at least one transformer layer"

		manifold_transformers = []
		for i in range(n_transformer_layers):
			if i == n_transformer_layers - 1:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out, norm_output=False, hidden_scale=1.5))
			else:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out, hidden_scale=1.5))

		self.manifold_transformer = nn.ModuleList(manifold_transformers)
		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out)).float()
		self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is now with shape [Batch, #patch, #encoded_h, #encoded_h]
		
		mask_idxes = None
		x_input = None
		if mask_ratio > 0.0:
			x_input = x.clone()
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			x_input[mask_idxes] = spd_mask
		else:
			x_input = x.clone()

		x_input = x_input.reshape(batch_size, num_patches, self.encoded_out, self.encoded_out).clone()
		for transformer in self.manifold_transformer:
			x_input = transformer(x_input, batch_size, num_patches)
		return x_input, mask_idxes
