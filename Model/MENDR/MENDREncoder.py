import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor

'''
Wavelet Encoder for MENDR with a decoder (only used for training)
Each frequency band has its own embedder, GAT, and decoder
'''
class WaveletEncoderDecoder(nn.Module):
    def __init__(self, num_channels, conv_kernel_size, conv_kernel_stride, seq_len, heads, encoded_h, decoder_size, decoder_stride, device):
        super(WaveletEncoderDecoder, self).__init__()
        assert len(decoder_size) == len(decoder_stride)
        self.num_channels = num_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_kernel_stride = conv_kernel_stride
        self.seq_len = seq_len
        self.heads = heads
        self.encoded_h = encoded_h
        self.device = device
        
        self.patch_embedder = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, 
            self.conv_kernel_size, stride=self.conv_kernel_stride,
            padding=self.conv_kernel_size//2),
            nn.Dropout1d(0.1),
            nn.GroupNorm(1, self.num_channels), # Same as Layer Norm
            nn.GELU()
        ).to(self.device)

        L_out = self.seq_len + 2 * (self.conv_kernel_size//2) - 1 * (self.conv_kernel_size - 1) - 1
        L_out = floor(L_out / self.conv_kernel_stride) + 1

        self.gnn_embedder = GATConv(L_out, self.encoded_h, heads=self.heads).to(self.device)
        self.gnn_group_norm = nn.GroupNorm(1, self.num_channels).to(self.device)
        self.gnn_dropout = nn.Dropout(p=0.1).to(self.device)
        self.gnn_gelu = nn.GELU().to(self.device)
        L_out = self.encoded_h * self.heads
        self.decoders = nn.Sequential()
        for i, (w, s) in enumerate(zip(decoder_size, decoder_stride)):
            if i == len(decoder_size) - 1:
                output_padding = self.seq_len - L_out
                self.decoders.add_module("Decoder_Final".format(i),
                nn.Sequential(
                    nn.ConvTranspose1d(self.num_channels, self.num_channels,
                    w, stride=s, padding=w//2, output_padding=0 if output_padding < 0 else output_padding),
                    nn.Dropout(0.1),
                    nn.GroupNorm(1, self.num_channels),
                    nn.GELU()
                    )
                )
                if output_padding < 0:
                    L_out = (L_out - 1) * s - 2 * (w//2) + 1 * (w - 1) + 1
                    self.decoders.add_module("Decoder_Final_Linear".format(i), nn.Linear(L_out, self.seq_len))
            else:
                self.decoders.add_module("Decoder_{}".format(i),
                nn.Sequential(
                    nn.ConvTranspose1d(self.num_channels, self.num_channels,
                    w, stride=s, padding=w//2),
                    nn.Dropout(0.1),
                    nn.GroupNorm(1, self.num_channels),
                    nn.GELU()
                    )
                )
                L_out = (L_out - 1) * s - 2 * (w//2) + 1 * (w - 1) + 1

        self.decoders = self.decoders.to(self.device)


    def forward(self, graph, x):
        patch_embedding = self.patch_embedder(x)

        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)

        patch_embedding = patch_embedding.view(-1, patch_embedding.shape[-1])

        encoding = self.gnn_embedder(patch_embedding, edge_index, edge_dist)
        encoding = encoding.view(-1, self.num_channels, self.encoded_h * self.heads)
        encoding = self.gnn_dropout(self.gnn_group_norm(encoding))
        encoding = self.gnn_gelu(encoding)

        decoding = self.decoders(encoding)

        return encoding, decoding

'''
Initialize Encoders for each wavelet band and 
put them into a single object.
'''
class MENDREncoder(nn.Module):
    def __init__(self, device):
        super(MENDREncoder, self).__init__()

        self.downsampling_factor = 0

        self.encoder_decoders = nn.ParameterDict({
            'delta': WaveletEncoderDecoder(
                    num_channels = 19,
                    conv_kernel_size = 2,
                    conv_kernel_stride = 2,
                    seq_len = 246,
                    heads = 4,
                    encoded_h = 120,
                    decoder_size = (2, 2, 1),
                    decoder_stride = (2, 2, 1),
                    device = device
                    ),

            'theta': WaveletEncoderDecoder(
                    num_channels = 19,
                    conv_kernel_size = 2,
                    conv_kernel_stride = 2,
                    seq_len = 246,
                    heads = 4,
                    encoded_h = 120,
                    decoder_size = (2, 2, 1),
                    decoder_stride = (2, 2, 1),
                    device = device
                    ),

            'alpha': WaveletEncoderDecoder(
                    num_channels = 19,
                    conv_kernel_size = 2,
                    conv_kernel_stride = 2,
                    seq_len = 486,
                    heads = 4,
                    encoded_h = 240,
                    decoder_size = (2, 2, 1),
                    decoder_stride = (2, 2, 1),
                    device = device
                    ),

            'beta': WaveletEncoderDecoder(
                    num_channels = 19,
                    conv_kernel_size = 2,
                    conv_kernel_stride = 2,
                    seq_len = 966,
                    heads = 4,
                    encoded_h = 480,
                    decoder_size = (2, 2, 1),
                    decoder_stride = (2, 2, 1),
                    device = device
                    ),

            'gamma': WaveletEncoderDecoder(
                    num_channels = 19,
                    conv_kernel_size = 2,
                    conv_kernel_stride = 2,
                    seq_len = 1925,
                    heads = 4,
                    encoded_h = 960,
                    decoder_size = (2, 2, 1),
                    decoder_stride = (2, 2, 1),
                    device = device
                    ),
        })

    def forward(self, graph, data):
        assert data.keys() == self.encoder_decoders.keys()
        output = {}
        for band, band_decomposition in data.items():
            output[band] = self.encoder_decoders[band](graph, band_decomposition)
        return output