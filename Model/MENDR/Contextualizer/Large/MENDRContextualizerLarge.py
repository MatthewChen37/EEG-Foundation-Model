import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from .MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes
from ..ManifoldTransformer import ManifoldTransformer

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
				patch_len,
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

		self.patch_len = patch_len
		self.encoded_out = encoded_out
		self.WaveletContextualizer = MENDRWaveletContextualizer(device=self.device, encoded_h=self.encoded_h, encoded_out=self.encoded_out, patch_len=self.patch_len)
		self.CombinedContextualizer = MENDRCombinedContextualizer(device=self.device, encoded_out=self.encoded_out)

		# Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(temp, requires_grad=True), requires_grad=True)

	def forward(self, x, batch_size, patch_num):
		wavelet_manifold_output, epoched_shape = self.WaveletContextualizer(x)
		combined_manifold_output = self.CombinedContextualizer._wavelet_LogEuclideanMean(wavelet_manifold_output)
		combined_manifold_output, mask_idxes = self.CombinedContextualizer(combined_manifold_output, epoched_shape, mask_ratio=0.0)
		# Never mask when calling it from here
		return combined_manifold_output, wavelet_manifold_output, mask_idxes # Adding this for consistency of API

class MENDRWaveletContextualizer(nn.Module):
	def __init__(self, device, encoded_h, encoded_out, patch_len=18, n_transformer_layers=2):
		super().__init__()
		self.device = device
		self.encoded_h = encoded_h
		self.encoded_out = encoded_out
		self.patch_len = patch_len

		self.learnable_padding = {
			'delta': None,
			'theta': None,
			'alpha': None,
			'beta': None,
			'gamma': None
		}

		self.learnable_padding = nn.ParameterDict(self.learnable_padding)

		for band in self.encoded_h:
			if self.encoded_h[band] and self.encoded_h[band] % 2 == 0:
				self.encoded_h[band] += 1
				print(f"Encoded_H {band} {self.encoded_h[band] - 1} is even, adding 1 to make it odd")
				self.learnable_padding[band] = torch.nn.Parameter(torch.zeros(1, 1, 1, self.patch_len), requires_grad=True).to(self.device)

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

		self.wavelet_manifold_transformers = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_manifold_transformers[band] = []
				for i in range(n_transformer_layers):
					if i == n_transformer_layers - 1:
						self.wavelet_manifold_transformers[band].append(ManifoldTransformer(device, self.encoded_out, norm_output=False, hidden_scale=1.5))
					else:
						self.wavelet_manifold_transformers[band].append(ManifoldTransformer(device, self.encoded_out, hidden_scale=1.5))
				self.wavelet_manifold_transformers[band] = nn.ModuleList(self.wavelet_manifold_transformers[band])
		self.wavelet_manifold_transformers = nn.ParameterDict(self.wavelet_manifold_transformers).to(self.device)

		self.wavelet_tangent_space = dict()
		for band in self.encoded_h:
			if self.encoded_h[band]:
				self.wavelet_tangent_space[band] = SPDTangentSpace(self.encoded_out, self.device)
		self.wavelet_tangent_space = nn.ParameterDict(self.wavelet_tangent_space).to(self.device)

	def forward(self, x):
		#assert x.keys() == self.wavelet_attention_manifolds.keys()
		# Batch Size, Num of Channels, Time Length
		x_input = dict()

		batch_size = x['delta'].shape[0]
		num_patches = x['delta'].shape[1]

		for band in x.keys():
			input_x = x[band].clone()
			if self.learnable_padding[band] is not None:
				input_x = torch.cat([input_x, self.learnable_padding[band].repeat(batch_size, num_patches, 1, 1)], dim=2)
			x_input[band] = input_x + self.position_encoder[band](input_x)

		wavelet_manifold_output = dict()
		for band, band_encodings in x_input.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings)
			cov_dim = wavelet_manifold_output[band].shape[2]
			#assert torch.allclose(wavelet_manifold_output[band], wavelet_manifold_output[band].mT, atol=(10 ** -10))
			wavelet_manifold_output[band] = wavelet_manifold_output[band].reshape(batch_size*num_patches, cov_dim, cov_dim)
			wavelet_manifold_output[band] = self.pre_attention_spd_transform[band](wavelet_manifold_output[band])
			wavelet_manifold_output[band] = wavelet_manifold_output[band].view(batch_size, num_patches, self.encoded_out, self.encoded_out)
			for transformer in self.wavelet_manifold_transformers[band]:
				wavelet_manifold_output[band] = transformer(wavelet_manifold_output[band], batch_size, num_patches)
				# output shape is [B, P, N, N]
		return wavelet_manifold_output, (batch_size, num_patches, self.encoded_out, self.encoded_out)

	def _batch_LogEuclideanMean(self, x, band):
		# X is list of [Batch_Size * epochs, C, C]
		x_stacked = torch.stack(x, dim=1)
		x_log = self.wavelet_manifold_transformers[band][0].manifold_self_attention.tensor_log(x_stacked)
		x_mean = self.wavelet_manifold_transformers[band][0].manifold_self_attention.tensor_exp(x_log.sum(dim=1, keepdim=True) / x_stacked.shape[1])[:, 0, :, :]
		return x_mean

class MENDRCombinedContextualizer(nn.Module):
	def __init__(self, device, encoded_out, n_transformer_layers=4):
		super().__init__()
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

	def forward(self, combined_manifold_output, og_output_shape, mask_ratio=0.0):
		# Combined Manifold Output should be a clone
		batch_size = og_output_shape[0]
		num_patches = og_output_shape[1]
		combined_manifold_output_hidden_dim = combined_manifold_output.shape[-1]
		mask_idxes = None
		x = None
		if mask_ratio > 0.0:
			x = combined_manifold_output.clone() # Just in case
			# Construct the mask at runtime
			spd_mask = torch.matmul(self.mask, self.mask.T)
			combined_manifold_output = combined_manifold_output.view(batch_size, num_patches, combined_manifold_output.shape[2], combined_manifold_output.shape[3])
			# We randomly mask each patch with probability mask_ratio
			# and calculate the LEM and then compare it with the full LEM
			# [B, P, C, C]
			mask_idxes = _make_mask_idxes(batch_size, num_patches, mask_ratio)
			combined_manifold_output[mask_idxes] = spd_mask

			'''
			print(mask_idxes.shape)
			for batch_idx in range(combined_manifold_output.shape[0]):
				for patch_idx in range(combined_manifold_output.shape[1]):
					if mask_idxes[batch_idx, patch_idx] == False:
						print("HERE", mask_idxes.shape)
						assert not torch.equal(combined_manifold_output[batch_idx, patch_idx], torch.zeros(combined_manifold_output[batch_idx, patch_idx].shape, device=self.device)), f"Masked SPD Matrix {batch_idx, patch_idx} is not equal to the mask"
			'''
		else:
			x = combined_manifold_output.clone()
		for transformer in self.manifold_transformer:
			x = transformer(x, batch_size, num_patches)
		return x, mask_idxes

	def _wavelet_LogEuclideanMean(self, x):
		x_input = dict()
		if len(x['delta'].shape) == 4:
			for band in x.keys():
				x_input[band] = x[band].clone().view(x[band].shape[0]*x[band].shape[1], x[band].shape[2], x[band].shape[3])
		else:
			for band in x.keys():
				x_input[band] = x[band].clone()
		combined_manifold_output = torch.stack(list(x_input.values()), dim=1)
		# Combined Manifold Output is something like [Batch_num * Patches, # of Wavelet Bands, C, C]
		combined_manifold_output = self.manifold_transformer[0].manifold_self_attention.tensor_log(combined_manifold_output)
		combined_manifold_output = self.manifold_transformer[0].manifold_self_attention.tensor_exp((combined_manifold_output.sum(dim=1, keepdim=True)) / combined_manifold_output.shape[1])
		return combined_manifold_output