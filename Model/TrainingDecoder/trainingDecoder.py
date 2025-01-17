import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil, floor

from types import SimpleNamespace
from ..MENDR.SpatialTemporalLayers import TimesBlock


'''
The reverse of ConvEncoder in encoder.py
Instead of downsampling, we upsample the input + TimesBlock.

Note that dec_width and dec_upsample should be 
in the reverse order of the encoder's width and downsample.
'''
class ConvDecoder(nn.Module):
	def __init__(self, encoder_h, out_features, dec_width, dropout, dec_upsample, original_time_len, top_k, num_kernels, device="cpu"):
		super().__init__()
		self.encoder_h = encoder_h
		self.out_features = out_features
		self.device = device

		# For times block
		self.original_time_len = original_time_len
		self.top_k = top_k
		self.num_kernels = num_kernels

		if not isinstance(dec_width, (list, tuple)):
			dec_width = [dec_width]
		if not isinstance(dec_upsample, (list, tuple)):
			dec_upsample = [dec_upsample]
		assert len(dec_upsample) == len(dec_width)

		# Centerable convolutions make life simpler
		dec_width = [e if e % 2 else e+1 for e in dec_width]
		self._upsampling = dec_upsample
		self._width = dec_width

		self.decoder = nn.Sequential()

		self.original_out_features = out_features
		if out_features % 2 != 0:
			'''
			if the number of out_features is odd, we add a "ghost" feature	
			'''
			out_features += 1

		for i, (width, upsample) in enumerate(zip(dec_width, dec_upsample)):
			self.decoder.add_module("Decoder_{}".format(i), nn.Sequential(
				nn.ConvTranspose1d(encoder_h, out_features, width, stride=upsample, padding=width // 2),
				nn.Dropout1d(dropout),
				nn.GroupNorm(out_features // 2, out_features),
				nn.GELU(),
			))
			encoder_h = out_features

		'''
		We project the output back to the original number of features 
		Helps when adding a "ghost" feature
		'''
		self.projection = nn.Conv1d(out_features, self.original_out_features, 1)

		# Calculate L_out from encoder
		enc_width = dec_width[::-1]
		enc_downsample = dec_upsample[::-1]
		L_out = original_time_len

		# Pass shape through encoder
		for w, s in zip(enc_width, enc_downsample):
			L_out = floor((L_out + 2 * (w // 2) - 1 * (width - 1) - 1) / s + 1)

		# Pass shape through projection
		L_out = floor((L_out + 2 * 0 - 1 * (1 - 1) - 1) / 1 + 1)

		# Pass through decoder
		for w, s in zip(dec_width, dec_upsample):
			L_out = (L_out - 1) * s - 2 * (w // 2) + 1 * (w - 1) + 0 + 1

		self.times_project = nn.Linear(L_out, self.original_time_len)
		self.times_block = TimesBlock(SimpleNamespace(seq_len=L_out, pred_len=self.original_time_len - L_out,
												  top_k=self.top_k, d_model=19, 
												  d_ff=self.encoder_h, num_kernels=self.num_kernels))

	def forward(self, x):
		x = self.decoder(x)
		x = self.projection(x)
		x = self.times_project(x)
		x = self.times_block(x.permute(0, 2, 1))
		x = x.permute(0, 2, 1)
		return x

	def load(self, filename, strict=True):
		state_dict = torch.load(filename)
		self.load_state_dict(state_dict, strict=strict)

	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def freeze_features(self, unfreeze=False):
		for param in self.parameters():
			param.requires_grad = unfreeze


if __name__ == "__main__":
	from encoder import ConvEncoder
	
	x = torch.randn(4, 19, 15360)

	encoder = ConvEncoder(in_features=19, encoder_h=256, enc_width=(3, 2, 2),
						  dropout=0., enc_downsample=(3, 2, 2))
	

	encoded_x = encoder(x)
	print("Encoded shape:", encoded_x.shape)

	decoder = ConvDecoder(encoder_h=256, out_features=19, dec_width=(2, 2, 3),
						  dropout=0., dec_upsample=(2, 2, 3), original_time_len=15360,
						  top_k=3, num_kernels=3)

	decoded_x = decoder(encoded_x)

	print("Decoded shape:", decoded_x.shape)

	assert decoded_x.shape == x.shape, f"Expected shape {x.shape}, got {decoded_x.shape}"

	print("All tests passed!")