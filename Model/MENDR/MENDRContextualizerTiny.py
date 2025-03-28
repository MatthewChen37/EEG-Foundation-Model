import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from .MENDRCommon import PositionalEncoding, _make_mask_idxes

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, device, encoded_h, patch_len=37, encoded_out=19):
		super(MENDRContextualizerTiny, self).__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_out = encoded_out
		self.patch_len = patch_len

		self.position_encoder = PositionalEncoding(self.device, self.encoded_h, self.patch_len, 0.1)
		self.e2r = E2R(device=self.device)
		self.ract = SPDRectified()
		self.pre_attention_transform = SPDTransform(self.encoded_h, self.encoded_out, self.device)
		self.Contextualizer = MENDRContextualizer(device=self.device, 
											encoded_out=encoded_out)

		
	def forward(self, x, mask_ratio=0.0):
		# x is a dict of wavelet bands of shape [Batch, #patch, #encoded_h, #time_step] 
		# Note that each wavelet band should be the same time length
		signal = []
		for band in x.keys():
			signal.append(x[band])
		signal = torch.cat(signal, dim=2).to(self.device)
		if self.position_encoder:
			signal = self.position_encoder(signal)
		cov_matrices = self.e2r(signal)
		batch_size = cov_matrices.shape[0]
		patches = cov_matrices.shape[1]
		signal_transformed = self.pre_attention_transform(self.ract(cov_matrices.reshape(batch_size*patches, self.encoded_h, self.encoded_h)))
		signal_transformed = signal_transformed.reshape(batch_size, patches, self.encoded_out, self.encoded_out)
		signal, mask_idxes = self.Contextualizer(signal_transformed, mask_ratio)
		return signal, signal_transformed, mask_idxes   # Return 2 things to keep compatibility with MENDRContextualizerLarge
	
class MENDRContextualizer(nn.Module):
	def __init__(self, device, encoded_out):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		self.encoded_out = encoded_out

		self.attention = AttentionManifold(self.encoded_out, self.encoded_out, self.device)
		self.spd_transform1 = SPDTransform(self.encoded_out, self.encoded_out, self.device)
		self.spd_transform2 = SPDTransform(self.encoded_out, self.encoded_out, self.device)
		self.ract = SPDRectified()

		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out))
		self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, x, mask_ratio=0.0):
		batch_size = x.shape[0]
		num_patches = x.shape[1]
		# x is now with shape [Batch, #patch, #encoded_h, #encoded_h]
		
		mask_idxes = None
		if mask_ratio > 0:
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			for batch_idx, masked_patch_idxes in enumerate(mask_idxes):
				for masked_epoch_idx in masked_patch_idxes:
					x[batch_idx, masked_epoch_idx, :, :] = spd_mask

		x, shape = self.attention(x)
		x = self.spd_transform1(x)
		x = self.ract(x)
		x = self.spd_transform2(x)
		x = x.reshape(shape[0], shape[1], self.encoded_out, self.encoded_out)
		return x, mask_idxes
	
	