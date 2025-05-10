import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from .MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes

'''
BENDR-style Contextualizer using mATT module 
'''
class MENDRContextualizerLarge(nn.Module):
	def __init__(self, device,
				delta_encoded_h,
                theta_encoded_h,
                alpha_encoded_h,
                beta_encoded_h,
                gamma_encoded_h,
                high_encoded_h,
				temp,
				encoded_out = 19):
		super().__init__()
		self.device = device
		self.encoded_h = {
			'delta': delta_encoded_h,
			'theta': theta_encoded_h,
			'alpha': alpha_encoded_h,
			'beta': beta_encoded_h,
			'gamma': gamma_encoded_h,
			#'high': high_encoded_h
		}

		self.learnable_padding = {
			'delta': None,
			'theta': None,
			'alpha': None,
			'beta': None,
			'gamma': None
		}

		for band in self.encoded_h:
			if self.encoded_h[band] and self.encoded_h[band] % 2 == 0:
				self.encoded_h[band] += 1
				print(f"Encoded_H {band} {self.encoded_h[band] - 1} is even, adding 1 to make it odd")
				self.learnable_padding[band] = torch.nn.Parameter(torch.zeros(1, 1, 1, 37), requires_grad=True).to(self.device)


		self.learnable_padding = nn.ParameterDict(self.learnable_padding)

		self.encoded_out = encoded_out
		self.WaveletContextualizer = MENDRWaveletContextualizer(device=self.device, encoded_h=self.encoded_h, encoded_out=self.encoded_out, patch_len=37)
		self.CombinedContextualizer = MENDRCombinedContextualizer(device=self.device, encoded_out=self.encoded_out)

		# Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(temp, requires_grad=True), requires_grad=True)

	def forward(self, x, batch_size, patch_num):
		for band in self.encoded_h.keys():
			if band in x:
				if self.learnable_padding[band] is not None:
					x[band] = torch.cat([x[band], self.learnable_padding[band].repeat(batch_size, patch_num, 1, 1)], dim=2)
		wavelet_manifold_output, epoched_shape = self.WaveletContextualizer(x)
		combined_manifold_output, mask_idxes = self.CombinedContextualizer(wavelet_manifold_output, epoched_shape, mask_ratio=0.0)
		# Never mask when calling it from here

		for band in wavelet_manifold_output.keys():
			wavelet_manifold_output[band] = wavelet_manifold_output[band].reshape(epoched_shape[0], epoched_shape[1],
			wavelet_manifold_output[band].shape[1], wavelet_manifold_output[band].shape[2])
		combined_manifold_output = combined_manifold_output.reshape(epoched_shape[0], epoched_shape[1],
		combined_manifold_output.shape[1], combined_manifold_output.shape[2])

		return combined_manifold_output, wavelet_manifold_output, mask_idxes # Adding this for consistency of API

	
class MENDRWaveletContextualizer(nn.Module):
	def __init__(self, device, encoded_h, encoded_out, patch_len=18):
		super().__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_out = encoded_out
		self.patch_len = patch_len

		self.ract = SPDRectified()
		# Positional Encoding
		self.position_encoder = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.position_encoder[band] = PositionalEncoding(self.device, self.encoded_h[band], self.patch_len, dropout=0.1)
		self.position_encoder = nn.ParameterDict(self.position_encoder).to(self.device)

		self.wavelet_e2r = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_e2r[band] = E2R(device=self.device)
		self.wavelet_e2r = nn.ParameterDict(self.wavelet_e2r).to(self.device)

		self.pre_attention_spd_transform = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.pre_attention_spd_transform[band] = SPDTransform(self.encoded_h[band], self.encoded_out, self.device)
		self.pre_attention_spd_transform = nn.ParameterDict(self.pre_attention_spd_transform).to(self.device)

		self.wavelet_attention_manifolds = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_attention_manifolds[band] = AttentionManifold(self.encoded_out, self.encoded_out, self.device)
		self.wavelet_attention_manifolds = nn.ParameterDict(self.wavelet_attention_manifolds).to(self.device)

		# Second Attention Manifold
		self.wavelet_attention_manifolds2 = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_attention_manifolds2[band] = AttentionManifold(self.encoded_out, self.encoded_out, self.device)
		self.wavelet_attention_manifolds2 = nn.ParameterDict(self.wavelet_attention_manifolds2).to(self.device)

		self.trace_normalization = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.trace_normalization[band] = BatchTraceNormalization(self.device)
		self.trace_normalization = nn.ParameterDict(self.trace_normalization).to(self.device)

		self.wavelet_tangent_space = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_tangent_space[band] = SPDTangentSpace(self.encoded_out, self.device)
		self.wavelet_tangent_space = nn.ParameterDict(self.wavelet_tangent_space).to(self.device)

		# SPD Transformations
		self.wavelet_spd_transforms = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_spd_transforms[band] = nn.Sequential(SPDTransform(self.encoded_out, int(1.5 * self.encoded_out), self.device),
																SPDRectified(),
																SPDTransform(int(1.5 *self.encoded_out), self.encoded_out, self.device))
		self.wavelet_spd_transforms = nn.ParameterDict(self.wavelet_spd_transforms).to(self.device)

		# Second SPD Transformations
		self.wavelet_spd_transforms2 = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_spd_transforms2[band] = nn.Sequential(SPDTransform(self.encoded_out, int(1.5 * self.encoded_out), self.device),
																SPDRectified(),
																SPDTransform(int(1.5 *self.encoded_out), self.encoded_out, self.device))
		self.wavelet_spd_transforms2 = nn.ParameterDict(self.wavelet_spd_transforms2).to(self.device)


	def forward(self, x):
		#assert x.keys() == self.wavelet_attention_manifolds.keys()
		# Batch Size, Num of Channels, Time Length
		x_input = dict()

		for band in x.keys():
			input_x = x[band].clone()
			x_input[band] = input_x + self.position_encoder[band](input_x)

		epoched_shape = None
		wavelet_manifold_output = dict()
		for band, band_encodings in x_input.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings)
			batch_size = wavelet_manifold_output[band].shape[0]
			num_patches = wavelet_manifold_output[band].shape[1]
			cov_dim = wavelet_manifold_output[band].shape[2]
			#assert torch.allclose(wavelet_manifold_output[band], wavelet_manifold_output[band].mT, atol=(10 ** -10))
			wavelet_manifold_output[band] = wavelet_manifold_output[band].reshape(batch_size*num_patches, cov_dim, cov_dim)
			wavelet_manifold_output[band] = self.pre_attention_spd_transform[band](wavelet_manifold_output[band])
			res_output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band].view(batch_size, num_patches, self.encoded_out, self.encoded_out))

			#assert torch.allclose(output, output.mT, atol=(10 ** -7)), "Attention Manifold"
			# Skip Connection
			epoched_shape = shape
			og_output_shape = res_output.shape
			wavelet_manifold_output[band] = wavelet_manifold_output[band] + res_output.view(wavelet_manifold_output[band].shape)
			wavelet_manifold_output[band] = wavelet_manifold_output[band].view(og_output_shape)
			wavelet_manifold_output[band] = self.trace_normalization[band](wavelet_manifold_output[band])
			#assert torch.allclose(output, output.mT, atol=(10 ** -7))

			'''
			# TODO: SOMETHING IS WRONG HERE -- For some reason when I include this it makes every patch embedding (almost) the same
			# Another skip connection
			wavelet_manifold_output[band] = wavelet_manifold_output[band] + self.wavelet_spd_transforms[band](wavelet_manifold_output[band]) # Add and norm
			wavelet_manifold_output[band] = self.trace_normalization[band](wavelet_manifold_output[band])

			'''
			# Second layer
			res_output2, shape2 = self.wavelet_attention_manifolds2[band](wavelet_manifold_output[band].view(batch_size, num_patches, self.encoded_out, self.encoded_out))
			epoched_shape = shape2
			og_output_shape = res_output2.shape
			wavelet_manifold_output[band] = wavelet_manifold_output[band] + res_output2.view(wavelet_manifold_output[band].shape)
			wavelet_manifold_output[band] = wavelet_manifold_output[band].view(og_output_shape)
			wavelet_manifold_output[band] = self.trace_normalization[band](wavelet_manifold_output[band])

			# Another skip connection
			# # TODO: SOMETHING IS WRONG HERE TOO -- For some reason when I include this it makes every patch embedding (almost) the same
			# wavelet_manifold_output[band] = wavelet_manifold_output[band] + self.wavelet_spd_transforms2[band](wavelet_manifold_output[band].clone()) # Just add, no norm
			# output shape is [B, N, N]
		return wavelet_manifold_output, shape

	def _batch_LogEuclideanMean(self, x, band):
		# X is list of [Batch_Size * epochs, C, C]
		x_stacked = torch.stack(x, dim=1)
		x_log = self.wavelet_attention_manifolds[band].tensor_log(x_stacked)
		x_mean = self.wavelet_attention_manifolds[band].tensor_exp(x_log.sum(dim=1, keepdim=True) / x_stacked.shape[1])[:, 0, :, :]
		return x_mean

class MENDRCombinedContextualizer(nn.Module):
	def __init__(self, device, encoded_out):
		super().__init__()
		self.device = device
		self.encoded_out = encoded_out

		self.combined_attention = AttentionManifold(self.encoded_out, self.encoded_out, self.device)
		self.ract = SPDRectified()
		self.combined_spd_transform = nn.Sequential(SPDTransform(self.encoded_out, int(1.5*self.encoded_out), self.device),
													self.ract,
													SPDTransform(int(1.5*self.encoded_out), self.encoded_out, self.device))

		self.trace_normalization = BatchTraceNormalization(self.device)

		self.combined_attention2 = AttentionManifold(self.encoded_out, self.encoded_out, self.device)
		self.ract2 = SPDRectified()
		self.combined_spd_transform2 = nn.Sequential(SPDTransform(self.encoded_out, int(1.5*self.encoded_out), self.device),
													self.ract2,
													SPDTransform(int(1.5*self.encoded_out), self.encoded_out, self.device))

		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out)).float().to(self.device)
		self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, x, og_output_shape, mask_ratio=0.0):
		batch_size = og_output_shape[0]
		num_patches = og_output_shape[1]
		mask_idxes = None
		x_input = dict()
		if mask_ratio > 0.0:
			for band in x.keys():
				x_input[band] = x[band].clone()

			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			for band, spd_batch in x_input.items():
				x_input[band] = spd_batch.view(batch_size, num_patches, spd_batch.shape[1], spd_batch.shape[2]).clone()
			# We randomly mask each patch with probability mask_ratio
			# and calculate the LEM and then compare it with the full LEM
			# [B, P, C, C]
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			for band in x_input.keys():
				x_input[band][mask_idxes] = spd_mask
				x_input[band] = x_input[band].view(batch_size * num_patches, spd_batch.shape[1], spd_batch.shape[2])
		else:
			for band in x.keys(): # No need to clone
				x_input[band] = x[band]

		# Log Euclidean Mean
		combined_manifold_output = self._wavelet_LogEuclideanMean(x_input)
		combined_manifold_output = combined_manifold_output.view(og_output_shape[0], og_output_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])
		combined_manifold_output_res, shape = self.combined_attention(combined_manifold_output)

		# Add and norm
		combined_manifold_output = combined_manifold_output.view(combined_manifold_output_res.shape) + combined_manifold_output_res
		combined_manifold_output = self.trace_normalization(combined_manifold_output)
		# Skip Connection
		combined_manifold_output = combined_manifold_output + self.combined_spd_transform(combined_manifold_output) # Add and norm
		combined_manifold_output = self.trace_normalization(combined_manifold_output)

		# Second layer
		combined_manifold_output = combined_manifold_output.view(og_output_shape[0], og_output_shape[1], combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
		combined_manifold_output_res2, shape2 = self.combined_attention2(combined_manifold_output)

		# Add and norm
		combined_manifold_output = combined_manifold_output.view(combined_manifold_output_res2.shape) + combined_manifold_output_res2
		combined_manifold_output = self.trace_normalization(combined_manifold_output)
		# Skip Connection
		combined_manifold_output = combined_manifold_output + self.combined_spd_transform2(combined_manifold_output) # Just add, no norm

		return combined_manifold_output, mask_idxes

	def _wavelet_LogEuclideanMean(self, x):
		combined_manifold_output = torch.stack(list(x.values()), dim=1)
		combined_manifold_output = self.combined_attention.tensor_log(combined_manifold_output)
		combined_manifold_output = self.combined_attention.tensor_exp((combined_manifold_output.sum(dim=1, keepdim=True)) / combined_manifold_output.shape[1])
		return combined_manifold_output