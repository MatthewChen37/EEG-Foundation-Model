import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from MENDRCommon import PositionalEncoding

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, device, encoded_h=190, patch_len=18, encoded_ff=380):
		super(MENDRContextualizerTiny, self).__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_ff = encoded_ff
		self.patch_len = patch_len

		self.position_encoder = PositionalEncoding(self.encoded_h, self.patch_len, 0.1)
		self.e2r = E2R(device=self.device)
		self.Contextualizer = MENDRContextualizer(device=self.device, 
											encoded_h=self.encoded_h, 
											encoded_ff=self.encoded_ff)

	def forward(self, x):
		# x is a dict of wavelet bands of shape [Batch, #patch, #encoded_h, #time_step] 
		# Note that each wavelet band should be the same time length
		signal = []
		cov_matrices = dict()
		for band in x.keys():
			cov_matrices[band] = self.e2r(x[band])
			signal.append(x[band])
		signal = torch.cat(signal, dim=2).to(self.device)
		if self.position_encoder:
			signal = self.position_encoder(signal)
		signal = self.Contextualizer(signal)
		return signal, cov_matrices  # Return 2 things to keep compatibility with MENDRContextualizerLarge
	
class MENDRContextualizer(nn.Module):
	def __init__(self, device, encoded_h, encoded_ff):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_ff = encoded_ff

		self.e2r = E2R(device=self.device)
		self.attention = AttentionManifold(self.encoded_h, self.encoded_ff, self.device)
		self.spd_transform1 = SPDTransform(self.encoded_ff, self.encoded_ff, self.device)
		self.spd_transform2 = SPDTransform(self.encoded_ff, self.encoded_h, self.device)
		self.ract = SPDRectified()

	def forward(self, x, mask=None):
		# x is with shape [Batch, #patch, #encoded_h, #time_step]
		x = self.e2r(x)
		x, shape = self.attention(x)
		x = self.ract(x)
		x = self.spd_transform1(x)
		x = self.ract(x)
		x = self.spd_transform2(x)
		return x 