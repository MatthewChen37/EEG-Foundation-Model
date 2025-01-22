import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil, floor

from types import SimpleNamespace

'''
The reverse of ConvEncoder in encoder.py
Instead of downsampling, we upsample the input + TimesBlock.

Note that dec_width and dec_upsample should be 
in the reverse order of the encoder's width and downsample.
'''
class ConvDecoder(nn.Module):
	def __init__(self, encoder_h, out_features, enc_width, enc_downsample, original_time_len):
		super().__init__()
		self.encoder_h = encoder_h
		self.out_features = out_features
		self.original_time_len = original_time_len

		# Calculate L_out from encoder
		L_out = self.original_time_len

		enc_width = [e if e % 2 else e+1 for e in enc_width]

		# Pass shape through encoder
		for w, s in zip(enc_width, enc_downsample):
			L_out = floor((L_out + 2 * (w // 2) - 1 * (w - 1) - 1) / s + 1)

		# To account for ConvTranspose1d, doesn't really do anything but just 
		# nice to acknowledge
		L_out = (L_out - 1) * 1 - 2 * (1 // 2) + 1 * (1 - 1) + 0 + 1

		self.L_out = L_out

		self.tanh = nn.Tanh()

		self.amplitude_decoder = nn.ConvTranspose1d(encoder_h, out_features, 1)
		self.amplitude_decoder_linear = nn.Sequential(nn.Linear(self.L_out, 2 * self.L_out),
													  nn.Tanh(),
													  nn.Linear(2 * self.L_out, self.original_time_len))
							   		
		self.angle_decoder = nn.ConvTranspose1d(encoder_h, out_features, 1)
		self.angle_decoder_linear = nn.Sequential(nn.Linear(self.L_out, 2 * self.L_out),
												  nn.Tanh(),
												  nn.Linear(2 * self.L_out, self.original_time_len))		
		
	def forward(self, x):
		B, T, N = x.shape
		# Ampltiude
		amplitude = self.amplitude_decoder(x)
		ampltiude = self.tanh(amplitude)
		ampltiude = ampltiude.view(-1, self.L_out)
		amplitude = self.amplitude_decoder_linear(amplitude)
		# Angle
		angle = self.angle_decoder(x)
		angle = self.tanh(angle)
		angle = angle.view(-1, self.L_out)
		angle = self.angle_decoder_linear(angle)
		return amplitude.view(B, self.out_features, self.original_time_len), angle.view(B, self.out_features, self.original_time_len)

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

	enc_width = (3, 2, 2)
	enc_downsample = (3, 2, 2)


	out = torch.fft.fft(x, dim=-1)
	amplitude = torch.real(out)
	angle = torch.angle(out)

	encoder = ConvEncoder(in_features=19, encoder_h=256, enc_width=enc_width, dropout=0.1, enc_downsample=enc_downsample)
	
	encoded_x = encoder(x)
	print("Encoded shape:", encoded_x.shape)

	decoder = ConvDecoder(encoder_h=256, out_features=19, enc_width=enc_width, enc_downsample=enc_downsample, original_time_len=x.shape[-1])

	decoded_amplitude, decoded_angle = decoder(encoded_x)

	print("Decoded shape:", decoded_amplitude.shape, decoded_angle.shape)

	assert decoded_amplitude.shape == amplitude.shape, f"Expected shape {amplitude.shape} but got {decoded_amplitude.shape}"
	assert decoded_angle.shape == angle.shape, f"Expected shape {angle.shape} but got {decoded_angle.shape}"

	print("All tests passed!")