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
				encoded_out = 19):
		super().__init__()
		self.device = device
		self.encoded_h = {
			'delta': delta_encoded_h,
			'theta': theta_encoded_h,
			'alpha': alpha_encoded_h,
			'beta': beta_encoded_h,
			'gamma': gamma_encoded_h,
			'high': high_encoded_h
		}
		self.encoded_out = encoded_out

		self.WaveletContextualizer = MENDRWaveletContextualizer(device=self.device, encoded_h=self.encoded_h, encoded_out=self.encoded_out, patch_len=37)
		self.CombinedContextualizer = MENDRCombinedContextualizer(device=self.device, encoded_out=self.encoded_out)

	def forward(self, x, batch_size, patch_num):
		# Reshape encodings before passing into contextualizer
		x_reshaped = dict()
		for band in self.encoded_h.keys():
			if band in x:
				x_reshaped[band] = x[band].clone().reshape(batch_size, patch_num, self.encoded_h[band], -1)
		wavelet_manifold_output, epoched_shape = self.WaveletContextualizer(x_reshaped)
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

		self.trace_normalization = BatchTraceNormalization(self.device)

		# SPD Transformations
		self.wavelet_spd_transforms = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_spd_transforms[band] = nn.Sequential(SPDTransform(self.encoded_out, int(1.5 * self.encoded_out), self.device),
																self.ract,
																SPDTransform(int(1.5 *self.encoded_out), self.encoded_out, self.device))
		self.wavelet_spd_transforms = nn.ParameterDict(self.wavelet_spd_transforms).to(self.device)


	def forward(self, x):
		#assert x.keys() == self.wavelet_attention_manifolds.keys()
		# Batch Size, Num of Channels, Time Length
		x_input = dict()
		if self.position_encoder:
			for band in x.keys():
					x_input[band] = self.position_encoder[band](x[band])

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
			res_output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band].clone().view(batch_size, num_patches, self.encoded_out, self.encoded_out))
			#assert torch.allclose(output, output.mT, atol=(10 ** -7)), "Attention Manifold"
			# Skip Connection
			epoched_shape = shape
			og_output_shape = res_output.shape
			wavelet_manifold_output[band] += res_output.view(wavelet_manifold_output[band].shape)
			wavelet_manifold_output[band] = wavelet_manifold_output[band].view(og_output_shape)
			wavelet_manifold_output[band] = self.trace_normalization(wavelet_manifold_output[band])
			#assert torch.allclose(output, output.mT, atol=(10 ** -7))
			# Another skip connection
			wavelet_manifold_output[band] += self.wavelet_spd_transforms[band](wavelet_manifold_output[band].clone())
			wavelet_manifold_output[band] = self.trace_normalization(wavelet_manifold_output[band])
		return wavelet_manifold_output, shape

	def _batch_LogEuclideanMean(self, x, band):
		# X is list of [Batch_Size * epochs, C, C]
		x = torch.stack(x, dim=1)
		x_log = self.wavelet_attention_manifolds[band].tensor_log(x)
		x_mean = self.wavelet_attention_manifolds[band].tensor_exp(x_log.sum(dim=1, keepdim=True) / x.shape[1])[:, 0, :, :]
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

		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(self.encoded_out, self.encoded_out))
		self.mask = nn.Parameter(self.mask, requires_grad=True)

	def forward(self, x, og_output_shape, mask_ratio=0.0):
		batch_size = og_output_shape[0]
		num_patches = og_output_shape[1]
		mask_idxes = None
		if mask_ratio > 0:
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			for band, spd_batch in x.items():
				x[band] = spd_batch.clone().view(batch_size, num_patches, spd_batch.shape[1], spd_batch.shape[2])

			# We randomly mask each patch with probability mask_ratio
			# and calculate the LEM and then compare it with the full LEM
			# [B, P, C, C]
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			for batch_idx, masked_patch_idxes in enumerate(mask_idxes):
				for band in x.keys():
					for masked_epoch_idx in masked_patch_idxes:
						x[band][batch_idx, masked_epoch_idx, :, :] = spd_mask

			for band in x.keys():
				x[band] = x[band].view(batch_size * num_patches, spd_batch.shape[1], spd_batch.shape[2])

		# Log Euclidean Mean
		combined_manifold_output = self._wavelet_LogEuclideanMean(x)
		combined_manifold_output = combined_manifold_output.view(og_output_shape[0], og_output_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])
		combined_manifold_output_res = combined_manifold_output.clone()
		combined_manifold_output, shape = self.combined_attention(combined_manifold_output)

		# Add and norm
		combined_manifold_output = combined_manifold_output + combined_manifold_output_res.view(combined_manifold_output.shape)
		combined_manifold_output = self.trace_normalization(combined_manifold_output)

		# Skip Connection
		combined_manifold_output_res = combined_manifold_output.clone()
		combined_manifold_output += self.combined_spd_transform(combined_manifold_output) # Just add, no norm
		combined_manifold_output = combined_manifold_output_res + combined_manifold_output

		return combined_manifold_output, mask_idxes

	def _wavelet_LogEuclideanMean(self, x):
		combined_manifold_output = torch.stack(list(x.values()), dim=1)
		combined_manifold_output = self.combined_attention.tensor_log(combined_manifold_output)
		combined_manifold_output = self.combined_attention.tensor_exp((combined_manifold_output.sum(dim=1, keepdim=True)) / combined_manifold_output.shape[1])
		return combined_manifold_output