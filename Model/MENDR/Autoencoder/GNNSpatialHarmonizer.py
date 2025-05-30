import torch
from torch import nn
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor
from einops import rearrange

class GNNSpatialHarmonizer(nn.Module):
    def __init__(self, num_channels, num_features, n_gnn_transformer_layers, hidden_ratio, heads):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.n_gnn_transformer_layers = n_gnn_transformer_layers
        self.gnn_transformers = nn.ModuleList([GNNTransformer(num_features=self.num_features, num_channels=num_channels, 
                                                              hidden_ratio=hidden_ratio, heads=heads) for _ in range(self.n_gnn_transformer_layers)])

    def forward(self, x, edge_index, edge_dist, B, P, C):
        # x: [Batch Size*Patches, Channels, num_features]
        for gnn_transformer in self.gnn_transformers:
            x = gnn_transformer(x, edge_index, edge_dist, B, P, C)
        return x

class GNNTransformer(nn.Module):
    def __init__(self, num_features, num_channels, hidden_ratio, heads):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.heads = heads
        self.hidden_ratio = hidden_ratio
        self.act = nn.GELU()

        self.gnn_channel_encoder = GATConv(num_features, num_features, heads=self.heads, concat=False)
        self.layer_norm1 = nn.LayerNorm((self.num_channels, self.num_features))
        self.gnn_lin = nn.Sequential(self.act, nn.Linear(self.num_features, self.hidden_ratio * self.num_features),
                                     self.act, nn.Linear(self.hidden_ratio * self.num_features, self.num_features)
                                     )
        self.layer_norm2 = nn.LayerNorm((self.num_channels, self.num_features))


    def forward(self, x, edge_index, edge_dist, B, P, C):
        # x: [Batch Size * Patches, Channels, num_features]
        # self-loops exist for self-node-attention
        # edge_index: [2, (num_channels^2) * B]
        # edge_dist: [(num_channels^2) * B, 1]
        x = x.view(B, P, C, self.num_features)
        
        gnn_input = x.reshape(B * P * C, self.num_features)
        '''
        batch_patch_edge_index = []
        for patch_idx in range(P):
            first_row = edge_index[0] + patch_idx * B * C
            second_row = edge_index[1] + patch_idx * B * C
            patch_edges = torch.stack([first_row, second_row], dim=0).to(x.device)
            batch_patch_edge_index.append(patch_edges)
        batch_patch_edge_index = torch.cat(batch_patch_edge_index, dim=-1)
        '''

        # Replace the loop with vectorized operations
        # Create P copies of the edge_index
        batch_patch_edge_index = edge_index.unsqueeze(0).repeat(P, 1, 1)  # [P, 2, num_edges]

        # Create offsets for each patch
        offsets = torch.arange(P, device=edge_index.device).view(P, 1, 1) * (B * C)

        # Add offsets to each copy
        batch_patch_edge_index = batch_patch_edge_index + offsets

        # Reshape to concatenate all edge indices
        batch_patch_edge_index = batch_patch_edge_index.permute(1, 0, 2).reshape(2, -1)
        # it is now [2, P * B * C * C]

        # Vectorized edge distance creation
        batch_patch_edge_dist = edge_dist.repeat((P, 1))
        # this is just [P * B * C * C]

        # Apply GNN to all patches simultaneously
        channel_encoding = self.gnn_channel_encoder(gnn_input, batch_patch_edge_index, batch_patch_edge_dist)
        
        # Reshape back and add residual connection
        channel_encoding = rearrange(channel_encoding, '(B P C) F -> B P C F', B=B, P=P, C=C)
        x = x + channel_encoding

        x = rearrange(x, 'B P C F -> (B P) C F', B=B, P=P, C=C)
        # x: [Batch Size * Patches, Channels, self.num_features]
        x = self.layer_norm1(x)
        x = x + self.gnn_lin(x)
        x = self.layer_norm2(x)
        return x

    def _sequential_forward(self, x, edge_index, edge_dist, B, P, C):
        # x: [Batch Size * Patches, Channels, num_features]
        # edge_index: [2, (num_channels^2) * B]
        # edge_dist: [(num_channels^2) * B, 1]
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
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.dropped_indices = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        # input: [Batch Size, Patches, Channels, num_features]
        if not self.training:
            return input
        
        # Generate a random mask (0s and 1s) for dropout
        mask = (torch.rand((input.size(0), input.size(1), input.size(2))) > self.p).float().to(self.device)

        # Apply the mask to the input
        output = input * mask.unsqueeze(3)
        
        # Store the dropped channel indices
        self.dropped_indices = mask == 0
        return output

if __name__ == "__main__":
    def setup_test_data():
        """Setup test data and parameters"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test parameters
        B, P, C = 2, 2, 3
        num_features = 4
        
        # Create test input
        x = torch.randn(B * P, C, num_features, device=device)
        first_row = []
        for batch_offset in range(B):
            for i in range(C):
                for j in range(C):
                    first_row.append(i + batch_offset * C)
        second_row = []
        for batch_offset in range(B):
            for i in range(C):
                for j in range(C):
                    second_row.append(j + batch_offset * C)
        edge_index = torch.tensor([first_row, second_row], device=device)

        # Create edge distances
        edge_dist = torch.randn((edge_index.size(1), 1), device=device)
        assert edge_index.shape == torch.Size([2, B * C * C]), f"Edge Index Shape: {edge_index.shape}, should be: (2, {B * C * C})"
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_dist': edge_dist,
            'B': B, 'P': P, 'C': C,
            'num_features': num_features,
            'device': device
        }

    data = setup_test_data()

    model = GNNTransformer(
        num_features=data['num_features'],
        num_channels=data['C'],
        hidden_ratio=2,
        heads=1,
    ).to(data['device'])

    model.eval()

    # Test vectorized implementation
    with torch.no_grad():
        result_vectorized = model.forward(
            data['x'].clone(), 
            data['edge_index'].clone(), 
            data['edge_dist'].clone(), 
            data['B'], data['P'], data['C']
        )
        
    # Test original loop implementation (manually implemented)
    with torch.no_grad():
        result_original = model._sequential_forward(
            data['x'].clone(), 
            data['edge_index'].clone(), 
            data['edge_dist'].clone(), 
            data['B'], data['P'], data['C']
        )

    try: 
        # Compare results
        assert torch.allclose(result_vectorized, result_original, atol=1e-5), "Vectorized and original implementations should produce identical results"
    except AssertionError as e:
        torch.set_printoptions(profile="full", linewidth=1000)
        print("Vectorized: ", result_vectorized.shape, "Original: ", result_original.shape)
        print("Vectorized: ", result_vectorized)
        print("Original: ", result_original)
        raise e

    print("Test passed!")