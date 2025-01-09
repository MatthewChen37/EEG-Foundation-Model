import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from SpatialTemporalLayers import TimesBlock, GNNLayer
from torch_geometric.utils import unbatch

class SpatialTemporalEncoder(nn.Module):
	def __init__(self, configs):
		super(SpatialTemporalEncoder, self).__init__()
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.k = configs.top_k
		self.times_block = TimesBlock(configs)
		self.gnn_layer = GNNLayer(configs)

	def forward(self, data):
		# x: PyG DataBatch Object 
		x = torch.tensor(np.vstack(data.x)).float()
		edge_index = data.edge_index
		edge_dist = torch.flatten(torch.tensor(np.stack(data.edge_attr)), start_dim=0, end_dim=1).float()
		x = self.gnn_layer(x, edge_index, edge_dist)
		# Convert to x: [B, T, N]
		x = unbatch(x, data.batch)
		x = torch.stack(x).float()
		print(x.shape)
		#x = self.times_block(x)
		# x: [B, T, N]
		return x