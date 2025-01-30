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
				AttentionManifold(19, 64, self.device)
			),
			'theta': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 64, self.device)
			),
			'alpha': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 64, self.device)
			),
			'beta': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 64, self.device)
			),
			'gamma': nn.Sequential(
				E2R(epochs=2, device=self.device),
				AttentionManifold(19, 64, self.device)
			)
		})

		self.combined_attention = AttentionManifold(64, 64, self.device)
		self.ract2 = SPDRectified()
	
		
		'''
		# Initialize replacement vector with 0's
		self.mask_replacement = torch.nn.Parameter(torch.normal(0, self.in_features**(-0.5), size=(self.in_features,)),
                                                   requires_grad=True)
		'''
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


		combined_wavelet_spd = torch.zeros(embedding_shapes['delta'][0], 2, 64, 64).to(self.device)
		for band in wavelet_output.keys():
			output = wavelet_output[band][0]
			shape  = wavelet_output[band][1]
			output = output.view((shape[0], shape[1], 64, 64))
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


def _make_mask(shape, p, total, span, allow_no_inds=False):
	# Note that shape is (batch_size, seq_len) and total = seq_len.
	# We do not care about the features because all features are masked, we only care about the time points.
	# TODO: This could probably be optimized
	mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

	# Iterate through each item in the batch.
	for i in range(shape[0]):
		mask_seeds = list()
		while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
			# For each time point, generate a random number and if it is less than p, add it to the mask seeds.
			# There is an index by 0 ([0]) because np.nonzero returns a tuple of arrays.
			# Mask seeds are the start indicies of the span.
			mask_seeds = np.nonzero(np.random.rand(total) < p)[0]
		spans = _make_span_from_seeds(mask_seeds, span, total=total)
		mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

	return mask

def _make_span_from_seeds(seeds, span, total=None):
	# TODO: This could probably be optimized
	inds = list()
	for seed in seeds:
		for i in range(seed, seed + span):
			# Break when i is greater than or equal to the number of time points.
			# Remember that the total is the number of time points and 
			# the range of the loop could be greater than the total.
			# This means that a mask is not always equal to span and could be less.
			if total is not None and i >= total:
				break
			elif i not in inds:
				# Add the index to the list of indices.
				inds.append(int(i))
	return np.array(inds)



if __name__ == "__main__":
	example_input = {
		'delta': (torch.randn(4, 19, 120), None),
		'theta': (torch.randn(4, 19, 120), None),
		'alpha': (torch.randn(4, 19, 240), None),
		'beta': (torch.randn(4, 19, 480), None),
		'gamma': (torch.randn(4, 19, 960), None)
	}

	contextualizer = MENDRContextualizer()
