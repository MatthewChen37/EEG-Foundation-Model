import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from ..layers import Permute, Flatten

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizer(nn.Module):
	def __init__(self, device):
		super(MENDRContextualizer, self).__init__()
		self.device = device
		
		self.wavelet_attention_manifolds = nn.ParameterDict({
			'delta': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'theta': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'alpha': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'beta': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 32, self.device)
			),
			'gamma': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 32, self.device)
			)
		})

		self.combined_attention = AttentionManifold(32, 32, self.device)
		self.ract2 = SPDRectified()
	
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.apply(self.init_bert_params)

	def forward(self, x, mask_t=None, mask_c=None):
		assert x.keys() == self.wavelet_attention_manifolds.keys()
		embedding_shapes = dict()
		for band in x.keys():
			# Batch Size, Num of Channels, Time Length
			embedding_shapes[band] = x[band][0].shape

		wavelet_output = dict()
		for band, band_encodings_decodings in x.items():
			wavelet_output[band] = self.wavelet_attention_manifolds[band](band_encodings_decodings[0])

		# The sum of SPD matrices is also SPD
		combined_wavelet_spd = torch.zeros(embedding_shapes['delta'][0], 2, 32, 32).to(self.device)
		for band in wavelet_output.keys():
			output = wavelet_output[band][0]
			shape  = wavelet_output[band][1]
			output = output.view((shape[0], shape[1], 32, 32))
			combined_wavelet_spd += output

		x, shape = self.combined_attention(combined_wavelet_spd)

		# TODO: FIX
		x = x.to(self.device)
		output = self.ract2(x)

		return output, shape, wavelet_output
	
	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False

	def init_bert_params(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight.data)
			if module.bias is not None:
				module.bias.data.zero_()
			# Tfixup
			module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data