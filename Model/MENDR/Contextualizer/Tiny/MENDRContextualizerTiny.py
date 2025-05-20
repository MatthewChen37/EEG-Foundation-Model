import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from .MENDRCommon import PositionalEncoding, _make_mask_idxes, BatchTraceNormalization
from ..ManifoldTransformer import ManifoldTransformer

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, device, encoded_h, patch_len=18, encoded_out=19):
		super(MENDRContextualizerTiny, self).__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_out = encoded_out
		self.patch_len = patch_len

		self.learnable_padding = None
		if self.encoded_h % 2 == 0:
			self.encoded_h += 1
			print(f"Encoded_H {self.encoded_h - 1} is even, adding 1 to make it odd")
			self.learnable_padding = nn.Parameter(torch.zeros(1, 1, 1, self.patch_len), requires_grad=True).to(self.device)

		self.position_encoder = PositionalEncoding(self.device, self.encoded_h, self.patch_len, 0.1)
		self.e2r = E2R(device=self.device)
		self.ract = SPDRectified()
		self.pre_attention_transform = SPDTransform(self.encoded_h, self.encoded_out, self.device)
		self.Contextualizer = MENDRContextualizer(device=self.device, 
											encoded_out=encoded_out)

		
	def forward(self, x, batch_size, num_patches, mask_ratio=0.0):
		# x is a dict of wavelet bands of shape [Batch, #patch, #encoded_h, #time_step] 
		# Note that each wavelet band should be the same time length
		signal = []
		for band in x.keys():
			signal.append(x[band])
		signal = torch.cat(signal, dim=2).to(self.device)
		if self.learnable_padding is not None:
			signal = torch.cat([signal, self.learnable_padding.repeat(batch_size, num_patches, 1, 1)], dim=2)
		if self.position_encoder:
			signal = signal + self.position_encoder(signal)
		cov_matrices = self.e2r(signal)
		signal_transformed = self.pre_attention_transform(cov_matrices.reshape(batch_size*num_patches, self.encoded_h, self.encoded_h))
		signal_transformed = signal_transformed.reshape(batch_size, num_patches, self.encoded_out, self.encoded_out)
		signal, mask_idxes = self.Contextualizer(signal_transformed.clone(), batch_size, num_patches, mask_ratio)
		return signal_transformed, signal, mask_idxes 
	
class MENDRContextualizer(nn.Module):
	def __init__(self, device, encoded_out, n_transformer_layers=3):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		self.encoded_out = encoded_out

		assert n_transformer_layers >= 1, "Must have at least one transformer layer"

		manifold_transformers = []
		for i in range(n_transformer_layers):
			if i == n_transformer_layers - 1:
				manifold_transformers.append(ManifoldTransformer(device, self.encoded_out, norm_output=False, hidden_scale=1.5))
			else:
				manifold_transformers.append(ManifoldTransformer(device, self.encoded_out, hidden_scale=1.5))

		self.manifold_transformer = nn.ModuleList(manifold_transformers)

		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out)).float().to(self.device)
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
	
	