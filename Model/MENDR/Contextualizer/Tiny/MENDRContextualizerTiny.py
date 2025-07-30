import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified, WaveletLogEuclideanMean
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.mAtt.utils import symmetric
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
		self.encoded_out = 19
		self.out_dim = out_dim
		self.include_high = include_high
		self.contextualizer_layers = contextualizer_layers
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
		self.Contextualizer = MENDRContextualizer(
											encoded_out=self.encoded_out,
											n_transformer_layers=self.contextualizer_layers)


		self.wavelet_conv_reduce = dict()
		for band in self.patch_lens.keys():
			self.wavelet_conv_reduce[band] = nn.Sequential(
												nn.Linear(self.out_dim*self.patch_lens[band], self.out_dim*self.patch_lens[band]),
												nn.GELU(),
												nn.Linear(self.out_dim*self.patch_lens[band], self.out_dim*self.patch_lens[band])
											)

		'''
		'''

		self.position_encoders = {
			'delta': PositionalEncoding(19, self.out_dim, self.patch_lens['delta']),
			'theta': PositionalEncoding(19, self.out_dim, self.patch_lens['theta']),
			'alpha': PositionalEncoding(19, self.out_dim, self.patch_lens['alpha']),
			'beta': PositionalEncoding(19, self.out_dim, self.patch_lens['beta']),
			'gamma': PositionalEncoding(19, self.out_dim, self.patch_lens['gamma']),
		}

		if self.include_high:
			self.position_encoders['high'] = PositionalEncoding(19, self.out_dim, self.patch_lens['high'])
			
		self.position_encoders = nn.ModuleDict(self.position_encoders)
		self.tangent_space = SPDTangentSpace(self.num_channels)
		self.wavelet_conv_reduce = nn.ModuleDict(self.wavelet_conv_reduce)

		self.wavelet_masks = dict()
		for band in self.patch_lens.keys():
			self.wavelet_masks[band] = torch.rand(19, out_dim*self.patch_lens[band])
		
	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is a dict of wavelet bands of shape [Batch, #patch, #num_channels, #time_step* #out_dim]
		# Each patch gets coagulated into the same time scale through the covariance matrix
		# Thus different patches will have different number of samples for calculating the covariance matrix
		x_og = dict()
		x_position_encodings = dict()
		x_masked = dict()

		for band in x.keys():
			x_og[band] = x[band].clone()
			x_og[band] = rearrange(x_og[band], 'B P C (O T) -> B C P (O T)', B=batch_size, P=num_patches, C=self.num_channels, O=self.out_dim, T=self.patch_lens[band])
			x_og[band] = self.wavelet_conv_reduce[band](x_og[band])
			x_og[band] = rearrange(x_og[band], 'B C P (O T) -> B P C (O T)', B=batch_size, P=num_patches, C=19, O=self.out_dim, T=self.patch_lens[band])
			x_masked[band] = x_og[band].clone()
			x_position_encodings[band] = self.position_encoders[band](x_og[band].clone())

		mask_idxes = None
		if mask_ratio > 0.0:
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			for band in x.keys():
				x_masked[band][mask_idxes] = self.wavelet_masks[band].to(x_og[band].device)
				x_og[band] = self.e2r(x_og[band])
		else:
			for band in x.keys():
				x_og[band] = self.e2r(x_og[band])

		for band in x.keys():
			x_masked[band] = x_masked[band] + x_position_encodings[band]
			x_masked[band] = self.e2r(x_masked[band])
		
		combined_manifold_output = WaveletLogEuclideanMean(x_og)
		combined_manifold_output_masked = WaveletLogEuclideanMean(x_masked)
		
		'''
		# this is an issue: https://github.com/arogozhnikov/einops/issues/204
		combined_manifold_output = rearrange(combined_manifold_output, 'B P C1 C2 -> (B P) C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
		# This is a transformation to generate a "token embedding" for each patch
		combined_manifold_output = self.pre_attention_transform(combined_manifold_output)
		combined_manifold_output = rearrange(combined_manifold_output, '(B P) C1 C2 -> B P C1 C2', B=batch_size, P=num_patches, C1=self.num_channels, C2=self.num_channels)
		'''
		combined_manifold_output_masked = self.Contextualizer(combined_manifold_output_masked, batch_size, num_patches, mask_ratio)
		return combined_manifold_output, combined_manifold_output_masked, mask_idxes, x_og
	
class MENDRContextualizer(nn.Module):
	def __init__(self, encoded_out, n_transformer_layers):
		super().__init__()
		self.encoded_out = encoded_out
		self.n_transformer_layers = n_transformer_layers

		assert n_transformer_layers >= 1, "Must have at least one transformer layer"

		manifold_transformers = []
		for i in range(n_transformer_layers):
			if i == n_transformer_layers - 1:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out, norm_output=False))
			else:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out))

		self.manifold_transformer = nn.ModuleList(manifold_transformers)
		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		# self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out)).float()
		#self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is now with shape [Batch, #patch, #encoded_h, #encoded_h]
		'''	
		mask_idxes = None
		x_input = None
		if mask_ratio > 0.0:
			x_input = x.clone()
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T).to(x.device)
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			x_input[mask_idxes] = spd_mask
		else:
			x_input = x.clone()
		'''
		#torch.set_printoptions(profile='full', linewidth=500, precision=6)
		x_input = x.reshape(batch_size, num_patches, self.encoded_out, self.encoded_out).clone()
		#print("\n")
		#print(f"Before Transformer: {x_input.shape}" + "*"*100)
		#print(x_input[0][0:5])
		for idx, transformer in enumerate(self.manifold_transformer):
			x_input = transformer(x_input, batch_size, num_patches)
			#print(f"{idx} Masked Embedding: {x_input.shape}", x_input[0][0:5])
		return x_input
