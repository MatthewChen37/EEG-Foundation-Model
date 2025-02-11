import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace
from ..layers import Permute, Flatten
import math

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizer(nn.Module):
	def __init__(self, device):
		super(MENDRContextualizer, self).__init__()
		self.device = device

		self.wavelet_e2r = nn.ParameterDict({
			'delta': E2R(epochs=4, device=self.device),
			'theta': E2R(epochs=4, device=self.device),
			'alpha': E2R(epochs=4, device=self.device),
			'beta': E2R(epochs=4, device=self.device),
			'gamma': E2R(epochs=4, device=self.device),
		})

		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': AttentionManifold(19, 19, self.device),
			'theta': AttentionManifold(19, 19, self.device),
			'alpha': AttentionManifold(19, 19, self.device),
			'beta': AttentionManifold(19, 19, self.device),
			'gamma': AttentionManifold(19, 19, self.device)
		})

		self.wavelet_r2e_tangent_spaces = nn.ParameterDict({
			'delta': SPDTangentSpace(19, self.device),
			'theta': SPDTangentSpace(19, self.device),
			'alpha': SPDTangentSpace(19, self.device),
			'beta': SPDTangentSpace(19, self.device),
			'gamma': SPDTangentSpace(19, self.device),
		})

		self.wavelet_r2e_lin = nn.ParameterDict({
			'delta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Dropout(p=0.1),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 380),
				nn.GELU(),
				nn.Linear(380, 190),
			),
			'theta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.Dropout(p=0.1),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 380),
				nn.GELU(),
				nn.Linear(380, 190),
			),
			'alpha': nn.Sequential(
				nn.Flatten(),
				nn.Dropout(p=0.1),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 380),
				nn.GELU(),
				nn.Linear(380, 190),
			),
			'beta': nn.Sequential(
				nn.Flatten(),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 380),
				nn.GELU(),
				nn.Linear(380, 190),
			),
			'gamma': nn.Sequential(
				nn.Flatten(),
				nn.Dropout(p=0.1),
				nn.GELU(),
				nn.GroupNorm(1, 760),
				nn.Linear(760, 380),
				nn.GELU(),
				nn.Linear(380, 190),
			)
		})

		self.combined_r2e_tangent_space = nn.Sequential(
			SPDTangentSpace(19, self.device)
		)

		self.combined_r2e_lin = nn.Sequential(
			nn.Flatten(),
			nn.GELU(),
			nn.Dropout(p=0.1),
			nn.GroupNorm(1, 760),
			nn.Linear(760, 380),
			nn.GELU(),
			nn.Linear(380, 380)
		)
		self.combined_attention = AttentionManifold(19, 19, self.device)
		self.ract = SPDRectified()

		self.apply(self.init_params)

		self.position_encoder = {
			'delta': PositionalEncoding(19, 0.1, 246),
			'theta': PositionalEncoding(19, 0.1, 246),
			'alpha': PositionalEncoding(19, 0.1, 486),
			'beta': PositionalEncoding(19, 0.1, 966),
			'gamma': PositionalEncoding(19, 0.1, 1925),
		}

	def forward(self, x):
		assert x.keys() == self.wavelet_attention_manifolds.keys()
		embedding_shapes = dict()
		for band in x.keys():
			# Batch Size, Num of Channels, Time Length
			embedding_shapes[band] = x[band][0].shape
			if self.position_encoder:
				x[band] = x.permute(2, 0, 1)
				x[band] = self.position_encoder[band](x[band])
				x[band] = x[band].permute(1, 2, 0)

		wavelet_manifold_output = dict()
		wavelet_r2e_output = dict()
		for band, band_encodings_decodings in x.items():
			wavelet_manifold_output[band] = self.wavelet_e2r[band](band_encodings_decodings[0])
			output, shape = self.wavelet_attention_manifolds[band](wavelet_manifold_output[band])
			# Skip Connection
			og_output_shape = output.shape
			output = output.view(wavelet_manifold_output[band].shape) + wavelet_manifold_output[band]
			output = output.view(og_output_shape) 
			wavelet_manifold_output[band] = (self.ract(output), shape)
			wavelet_r2e_output[band] = self.wavelet_r2e_tangent_spaces[band](output)
			wavelet_r2e_output[band] = wavelet_r2e_output[band].view(shape[0], shape[1], -1)
			wavelet_r2e_output[band] = self.wavelet_r2e_lin[band](wavelet_r2e_output[band])

		# The sum of SPD matrices is also SPD
		combined_manifold_output = torch.zeros(embedding_shapes['delta'][0], 4, 19, 19).to(self.device)
		for band in wavelet_manifold_output.keys():
			output = wavelet_manifold_output[band][0]
			shape  = wavelet_manifold_output[band][1]
			output = output.view((shape[0], shape[1], 19, 19))
			combined_manifold_output += output

		combined_manifold_output_skip, shape = self.combined_attention(combined_manifold_output)
		# Skip connection
		combined_manifold_output =  combined_manifold_output.view(shape[0]*shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3]) + combined_manifold_output_skip
		combined_r2e_output = self.combined_r2e_tangent_space(combined_manifold_output)
		combined_r2e_output = combined_r2e_output.view(shape[0], shape[1], -1)
		combined_r2e_output = self.combined_r2e_lin(combined_r2e_output)
		return combined_r2e_output, combined_manifold_output, wavelet_r2e_output, wavelet_manifold_output

	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False

	def init_params(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight.data)
			if module.bias is not None:
				module.bias.data.zero_()

# From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)