import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from math import ceil, floor

'''
Predict wavelet coefficients from encoder output

Delta: 0-4Hz
Theta: 4-8Hz
Alpha: 8-16Hz
Beta: 16-32Hz
Gamma: 32-64Hz
Other: 64-128 Hz
High: 128-256 Hz
'''
class WaveletDecoder(nn.Module):

	def __init__(self, encoder_h, out_features, enc_width, enc_downsample, original_time_len):
		super().__init__()
		self.encoder_h = encoder_h
		self.out_features = out_features
		self.original_time_len = original_time_len

		# Centerable convolutions are nicer
		enc_width = [e if e % 2 else e+1 for e in enc_width]
		L_out = self.original_time_len
		# Pass shape through encoder
		for w, s in zip(enc_width, enc_downsample):
			L_out = floor((L_out + 2 * (w // 2) - 1 * (w - 1) - 1) / s + 1)
		self.L_out = L_out

		# TODO: Currently these are hardcoded under the assumption the data 
		# is a one minute recording at 256Hz. 

		self.delta_decoder = self._initialize_decoder(246)
		self.theta_decoder = self._initialize_decoder(246)
		self.alpha_decoder = self._initialize_decoder(486)
		self.beta_decoder = self._initialize_decoder(966)
		self.gamma_decoder = self._initialize_decoder(1926)
		self.other_decoder = self._initialize_decoder(3845)
		self.high_decoder = self._initialize_decoder(7683)


	def _initialize_decoder(self, output_length):
		return nn.Sequential(
			nn.ConvTranspose1d(self.encoder_h, self.out_features * 4, kernel_size=4, stride=2),
			nn.GELU(),
			nn.GroupNorm((4 * self.out_features) // 2, self.out_features * 4),
			nn.ConvTranspose1d(self.out_features * 4, self.out_features * 2, kernel_size=2, stride=2),
			nn.GELU(),
			nn.GroupNorm((2 * self.out_features) // 2, self.out_features * 2),
			nn.ConvTranspose1d(self.out_features * 2, self.out_features, kernel_size=1, stride=3),
			nn.GELU(),
			nn.Linear(7690, output_length)
		)

	def forward(self, x):
		delta_coefficients = self.delta_decoder(x)
		theta_coefficients = self.theta_decoder(x)
		alpha_coefficients = self.alpha_decoder(x)
		beta_coefficients = self.beta_decoder(x)
		gamma_coefficients = self.gamma_decoder(x)
		other_coefficients = self.other_decoder(x)
		high_coefficients = self.high_decoder(x)

		return {
			'delta': delta_coefficients,
			'theta': theta_coefficients,
			'alpha': alpha_coefficients,
			'beta': beta_coefficients,
			'gamma': gamma_coefficients,
			'other': other_coefficients,
			'high': high_coefficients,
		}
	
	def load(self, filename, strict=True):
		state_dict = torch.load(filename)
		self.load_state_dict(state_dict, strict=strict)

	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def freeze_features(self, unfreeze=False):
		for param in self.parameters():
			param.requires_grad = unfreeze

if __name__ == "__main__":
	import sys
	sys.path.append("../")
	from encoder import ConvEncoder
	from WaveletLoss import WaveletLoss

	x = torch.randn(4, 19, 15360)

	enc_width = (3, 2, 2)
	enc_downsample = (3, 2, 2)

	encoder = ConvEncoder(in_features=19, encoder_h=128, enc_width=enc_width,
					    dropout=0.1, enc_downsample=enc_downsample)

	encoded_x = encoder(x)
	print("Encoded shape:", encoded_x.shape)

	decoder = WaveletDecoder(128, 19, enc_width=enc_width,
						   enc_downsample=enc_downsample, original_time_len=15360)


	recon_coefficients = decoder(encoded_x)

	expected_shapes = {
		'delta': (4, 19, 246),
		'theta': (4, 19, 246),
		'alpha': (4, 19, 486),
		'beta': (4, 19, 966),
		'gamma': (4, 19, 1926),
		'other': (4, 19, 3845),
		'high': (4, 19, 7683)
	}

	for band in recon_coefficients.keys():
		assert recon_coefficients[band].shape == expected_shapes[band]

	band_coeffs = {
		'delta': 1,
		'theta': 1,
		'alpha': 1,
		'beta': 1,
		'gamma': 1,
		'other': 1,
		'high': 1
	}

	loss = WaveletLoss(x, recon_coefficients, band_coeffs=band_coeffs)

	print("Loss: ", loss.item())
	print("All shapes match expected shapes.")