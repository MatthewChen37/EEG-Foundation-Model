import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from .SpatialTemporalLayers import TimesBlock, GNNLayer
from torch_geometric.utils import unbatch

class SpatialTemporalEncoder(nn.Module):
	def __init__(self, configs):
		super(SpatialTemporalEncoder, self).__init__()
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.k = configs.top_k
		self.times_block = TimesBlock(configs)
		self.gnn_layer = GNNLayer(configs)
		self.predict_linear = nn.Linear(configs.seq_len, configs.seq_len + configs.pred_len)
		self.project_back = nn.Linear(configs.seq_len + configs.pred_len, configs.seq_len)
	def forward(self, data):
		# x: PyG DataBatch Object 
		x = torch.tensor(np.vstack(data.x)).float()
		edge_index = data.edge_index
		edge_dist = torch.flatten(torch.tensor(np.stack(data.edge_attr)), start_dim=0, end_dim=1).float()
		x = self.gnn_layer(x, edge_index, edge_dist)
		# Convert to x: [B, N, T]
		x = unbatch(x, data.batch)
		# Permute to x: [B, T, N]
		x = torch.stack(x).float()
		# See https://github.com/thuml/Time-Series-Library/blob/cdf8f0c3c5e79c1e8152e71dc35009ae46a6a920/models/TimesNet.py#L113
		# for why this is done
		# x: [B, N, 2T]
		x = self.predict_linear(x)
		x = nn.functional.gelu(x) # Include some non-linearity
		# x: [B, T, N] # N here is the number of channels
		x = self.times_block(x.permute(0, 2, 1))
		# x: [B, N, T]
		x = self.project_back(x.permute(0, 2, 1))
		x = nn.functional.gelu(x)
		return x