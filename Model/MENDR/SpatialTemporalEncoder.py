import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from SpatialTemporalLayers import TimesBlock, GNNLayer

class SpatialTemporalEncoder(nn.Module):
	def __init__(self, configs):
		super(SpatialTemporalEncoder, self).__init__()
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.k = configs.top_k
		self.times_block = TimesBlock(configs)
		self.gnn_layer = GNNLayer(configs)

	def forward(self, x):
		# x: [B, T, N]
		x = self.times_block(x)
		x = self.gnn_layer(x)
		return x

