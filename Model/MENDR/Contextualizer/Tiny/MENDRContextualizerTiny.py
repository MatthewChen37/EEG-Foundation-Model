import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified, WaveletLogEuclideanMean
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import PositionalEncoding, _make_mask_idxes, BatchTraceNormalization, LogEuclidLayerNorm
from Model.MENDR.Contextualizer.ManifoldTransformer import ManifoldTransformer

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, num_channels, out_dim, patch_len, encoded_out, patch_lens=None,contextualizer_layers=3):
		super(MENDRContextualizerTiny, self).__init__()
		self.num_channels = num_channels
		self.out_dim = out_dim
		self.patch_len = patch_len
		self.encoded_out = encoded_out
		if patch_lens is None:
			self.patch_lens = {
				'delta': 8,
				'theta': 8,
				'alpha': 16,
				'beta': 32,
				'gamma': 64,
			}
		else:
			self.patch_lens = patch_lens

		self.e2r = E2R()
		self.ract = SPDRectified()
		self.pre_attention_transform = SPDTransform(self.num_channels, self.encoded_out)
		self.Contextualizer = MENDRContextualizer(
											encoded_out=self.encoded_out,
											n_transformer_layers=contextualizer_layers)

		self.position_encoders = {
			'delta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['delta']),
			'theta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['theta']),
			'alpha': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['alpha']),
			'beta': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['beta']),
			'gamma': PositionalEncoding(self.num_channels, self.out_dim, self.patch_lens['gamma']),
		}
		self.position_encoders = nn.ParameterDict(self.position_encoders)

	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is a dict of wavelet bands of shape [Batch, #patch, #num_channels*out_dim=encoded_h, #time_step]
		# Each patch gets coagulated into the same time scale through the covariance matrix
		# Thus different patches will have different number of samples for calculating the covariance matrix
		cov_matrices = dict()
		for band in x.keys():
			x[band] = self.position_encoders[band](x[band])
			cov_matrices[band] = self.e2r(x[band])
		combined_manifold_output = WaveletLogEuclideanMean(cov_matrices)
		print(combined_manifold_output.shape)
		#cov_matrices = self.e2r(signal)
		#signal_unmasked = self.pre_attention_transform(cov_matrices.reshape(batch_size*num_patches, self.encoded_h, self.encoded_h))

		'''
		signal_unmasked = signal_unmasked.reshape(batch_size, num_patches, self.encoded_out, self.encoded_out)
		signal_masked, mask_idxes = self.Contextualizer(signal_unmasked.clone(), batch_size, num_patches, mask_ratio)
		return signal_unmasked, signal_masked, mask_idxes 
		'''
	
class MENDRContextualizer(nn.Module):
	def __init__(self, encoded_out, n_transformer_layers=3):
		super(MENDRContextualizer, self).__init__()
		self.encoded_out = encoded_out

		assert n_transformer_layers >= 1, "Must have at least one transformer layer"

		manifold_transformers = []
		for i in range(n_transformer_layers):
			if i == n_transformer_layers - 1:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out, norm_output=False, hidden_scale=1.5))
			else:
				manifold_transformers.append(ManifoldTransformer(self.encoded_out, hidden_scale=1.5))

		self.manifold_transformer = nn.ModuleList(manifold_transformers)


		#self.log_euclid_layer_norm = LogEuclidLayerNorm(device, num_channels=self.encoded_out, epsilon=1e-5)
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


class EuclideanPositionalEncoding(nn.Module):
	def __init__(self, encoded_h, patch_len, dropout=0.1):
		super().__init__()
		self.encoded_h = encoded_h
		self.len = patch_len
		self.dropout = nn.Dropout(p=dropout)

		self.positional_encoders = nn.ParameterDict({
			'delta': PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
			'theta': PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
			'alpha': PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
			'beta':  PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
			'gamma': PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
			#'high': PositionalEncoding(self.encoded_h, self.len, dropout=dropout),
		})

