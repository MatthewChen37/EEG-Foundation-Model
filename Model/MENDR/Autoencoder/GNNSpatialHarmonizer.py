import torch
from torch import nn
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor
from Model.MENDR.MENDRCommon import Std_Norm

class GNNSpatialHarmonizer(nn.Module):
    def __init__(self, num_channels, num_features, device, n_gnn_transformer_layers=2, heads=1):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.n_gnn_transformer_layers = n_gnn_transformer_layers
        self.gnn_transformers = nn.ModuleList([GNNTransformer(num_features=self.num_features, num_channels=num_channels, 
                                               device=device, hidden_ratio=4, heads=heads) for _ in range(self.n_gnn_transformer_layers)])

        self.patch_normalizers = Std_Norm()

        self.dropout = Dropout1dWithIndexTracking(device, p=0.1)

    def forward(self, x, edge_index, edge_dist, B, P, C, dropout=False):
        # x: [Batch Size*Patches, Channels, num_features]
        if dropout:
            x = self.dropout(x.clone().reshape(B*P, C, self.num_features))
        for gnn_transformer in self.gnn_transformers:
            x = gnn_transformer(x, edge_index, edge_dist, B, P, C)

        if dropout:
            return x, self.dropout.dropped_indices
        else:
            return x

    def _super_patchify(self, data, super_patch_length, super_patch_hop_length):
        patched_wavelet_data = []
        wavelet_super_patch_length = super_patch_length
        wavelet_super_patch_hop_length = super_patch_hop_length
        for i in range(0, data.shape[-1], wavelet_super_patch_hop_length):
            if i + super_patch_length <= data.shape[-1]:
                patched_wavelet_data.append(data[:, :, i:i + wavelet_super_patch_length])
        patched_wavelet_data = torch.stack(patched_wavelet_data, dim=1) # [Batch, Patches, C, T]
        patched_wavelet_data = self.patch_normalizers(patched_wavelet_data)
        return patched_wavelet_data


class GNNTransformer(nn.Module):
    def __init__(self, num_features, num_channels, device, hidden_ratio=2, heads=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.heads = heads
        self.device = device
        self.hidden_ratio = hidden_ratio
        self.act = nn.GELU()

        self.gnn_channel_encoder = GATConv(num_features, num_features, heads=self.heads, concat=False).to(self.device)
        self.layer_norm1 = nn.LayerNorm((self.num_channels, self.num_features))
        self.gnn_lin = nn.Sequential(self.act, nn.Linear(self.num_features, self.hidden_ratio * self.num_features),
                                     self.act, nn.Linear(self.hidden_ratio * self.num_features, self.num_features)
                                     ).to(self.device)
        self.layer_norm2 = nn.LayerNorm((self.num_channels, self.num_features))

    def forward(self, x, edge_index, edge_dist, B, P, C):
        # x: [Batch Size*Patches, Channels, num_features]
        x = x.view(B, P, C, self.num_features)
        for patch_idx in range(x.shape[1]):
            gnn_channel_encoder_input = x[:, patch_idx, :, :].reshape(B * C, self.num_features).clone()
            # gnn_channel_encoder_input: [Batch Size * Channels (Each entry is a node), self.num_features]
            #print("Details:")
            #torch.set_printoptions(profile="full", linewidth=1000)
            #print(edge_index[0])
            #print(edge_index[1])
            #print(edge_dist.shape)
            #print(gnn_channel_encoder_input.shape, edge_index.min(), edge_index.max())
            channel_encoding = self.gnn_channel_encoder(gnn_channel_encoder_input, edge_index, edge_dist)
            # channel_encoding: [Batch Size, Channels, self.num_features]
            # assert x[:, patch_idx, :, :].shape == channel_encoding.reshape(B, C, self.num_features).shape
            x[:, patch_idx, :, :] = x[:, patch_idx, :, :] + channel_encoding.reshape(B, C, self.num_features)

        x = x.view(B*P, C, self.num_features)
        # x: [Batch Size * Patches, Channels, self.hidden_ratio * self.num_features]
        x = self.layer_norm1(x)
        x = x + self.gnn_lin(x)
        x = self.layer_norm2(x)
        return x

class Dropout1dWithIndexTracking(nn.Dropout1d):
    def __init__(self, device, p=0.5, inplace=False):
        super(Dropout1dWithIndexTracking, self).__init__(p, inplace)
        self.dropped_indices = None
        self.device = device

    def forward(self, input):
        if not self.training:
            return input
        
        # Generate a random mask (0s and 1s) for dropout
        mask = (torch.rand((input.size(0), input.size(1))) > self.p).float().to(self.device)

        # Apply the mask to the input
        output = input * mask.unsqueeze(2)
        
        # Store the dropped channel indices
        self.dropped_indices = mask == 0
        return output

