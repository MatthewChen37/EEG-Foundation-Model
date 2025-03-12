import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor


'''
We break each Wavelet sequence into patches. There are two patch scales: Super-Patch and Sub-Patch. 
By default super-patches represent 10 seconds and are what are first fed into the encoder.
The encoder then breaks the data up into smaller Sub-Patches (approx 1 sec by default).
Decoders are used for Super-Patch reconstruction to ensure embeddings have a real, physical
intepretation. 
'''
class WaveletEncoderDecoder(nn.Module):
    def __init__(self, num_channels, sub_patch_size, heads, super_patch_seq_len, encoded_h, device):
        super(WaveletEncoderDecoder, self).__init__()
        self.num_channels = num_channels
        self.channel_dropout = nn.Dropout1d(0.1)
        self.patch_size = sub_patch_size
        self.stride = self.patch_size // 2
        self.device = device
        self.heads = heads
        self.encoded_h = encoded_h
        self.seq_len = super_patch_seq_len

        self.act = nn.GELU()
        L_out = self.seq_len + 2 * (0) - 1 * ((self.patch_size) - 1) - 1
        L_out = floor(L_out / (self.stride)) + 1
        self.patch_embedder = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=self.patch_size, stride=self.stride, groups=1).to(self.device)
        self.patch_embedder_lin = nn.Sequential(self.act, nn.Linear(L_out, L_out))

        #self.gnn_channel_encoder = SplineConv(L_out, self.seq_len, dim=1, kernel_size=3).to(self.device)
        self.gnn_channel_encoder = GATConv(L_out, L_out, heads=self.heads, concat=False).to(self.device)
        self.gnn_lin1 = nn.Sequential(self.act, nn.Linear(L_out, L_out))

        self.layer_norm1 = nn.LayerNorm((19, L_out))
        self.layer_norm2 = nn.LayerNorm((19, L_out))

        L_out = L_out + 2 * (0) - 1 * (2 - 1) - 1
        L_out = floor(L_out / 1) + 1
        self.patch_embedder2 = nn.Conv1d(in_channels=19, out_channels=encoded_h, kernel_size=2, stride=1, groups=1, padding=0)
        self.patch_embedder2_lin = nn.Sequential(self.act, nn.Linear(L_out, L_out))

        # Decoders
        L_out = (L_out - 1) * self.stride - 2 * 0 + 1 * (1 - 1) + 0 + 1
        self.up1 = nn.ConvTranspose1d(in_channels=encoded_h, out_channels=encoded_h * 4, kernel_size=1, stride=self.stride, groups=1)
        L_out = (L_out - 1) * (1) - 2 * 0 + 1 * (1 - 1) + 0 + 1
        self.up2 = nn.ConvTranspose1d(in_channels=encoded_h * 4, out_channels=19, kernel_size=1, stride=1, groups=19)
        self.up3 = nn.Sequential(self.act, nn.Linear(L_out, self.seq_len))

    def getEncoderParamCount(self):
        patch_embedder_count = sum(p.numel() for p in self.patch_embedder.parameters() if p.requires_grad)
        gnn_encoder_count = sum(p.numel() for p in self.gnn_channel_encoder.parameters() if p.requires_grad)
        return patch_embedder_count + gnn_encoder_count

    def getDecoderParamCount(self):
        up1_count = sum(p.numel() for p in self.up1.parameters() if p.requires_grad)
        up2_count = sum(p.numel() for p in self.up2.parameters() if p.requires_grad)
        up3_count = sum(p.numel() for p in self.up3.parameters() if p.requires_grad)
        return up1_count + up2_count + up3_count

    def forward(self, graph, x):
        # x: [Batch Size, Channels, Time Steps]
        x = self.channel_dropout(x)
        x = self.patch_embedder(x)
        x = self.patch_embedder_lin(x)
        x = self.layer_norm1(x)
        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)
        gnn_channel_encoder_input1 = x.view(-1, x.shape[-1])
        channel_encoding = self.gnn_channel_encoder(gnn_channel_encoder_input1, edge_index, edge_dist)
        x = x + channel_encoding.view(x.shape[0], -1, channel_encoding.shape[-1])
        x = self.layer_norm2(x)
        x = self.gnn_lin1(x)
        x = self.patch_embedder2(x)
        x = self.patch_embedder2_lin(x)

        decoding = self.up1(x)
        decoding = self.channel_dropout(decoding)
        decoding = self.up2(decoding)
        decoding = self.up3(decoding)
        return x, decoding

'''
Initialize Encoders for each wavelet band and 
put them into a single object.
'''
class PatchEncoderDecoder(nn.Module):
    def __init__(self, device):
        super(PatchEncoderDecoder, self).__init__()
        self.device = device
        self.encoder_decoders = nn.ParameterDict({
            'delta': WaveletEncoderDecoder(
                    num_channels = 19,
                    sub_patch_size=4,
                    encoded_h=38,
                    heads=1,
                    super_patch_seq_len=41,
                    device = device
                    ),

            'theta': WaveletEncoderDecoder(
                    num_channels = 19,
                    sub_patch_size=4,
                    encoded_h=38,
                    heads=1,
                    super_patch_seq_len=41,
                    device = device
                    ),
            'alpha': WaveletEncoderDecoder(
                    num_channels = 19,
                    sub_patch_size=8,
                    encoded_h=38,
                    heads=1,
                    super_patch_seq_len=81,
                    device = device 
                    ),

            'beta': WaveletEncoderDecoder(
                    num_channels = 19,
                    sub_patch_size=16,
                    encoded_h=38,
                    heads=1,
                    super_patch_seq_len = 161,
                    device = device
                    ),

            'gamma': WaveletEncoderDecoder(
                    num_channels = 19,
                    sub_patch_size = 32,
                    encoded_h = 76,
                    heads=1,
                    super_patch_seq_len=320,
                    device = device
                    ),
        })

    def forward(self, graph, data):
        assert data.keys() == self.encoder_decoders.keys()
        output = {}
        for band, band_decomposition in data.items():
            output[band] = self.encoder_decoders[band](graph, band_decomposition)
        return output

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if finetuning:
            self.mask_replacement.requires_grad = False


'''
Wavelet Encoder for MENDR with a decoder (only used for training)
Each frequency band has its own embedder, GAT, and decoder
class WaveletEncoderDecoder(nn.Module):
    def __init__(self, num_channels, seq_len, heads, encoder_conv_kernel_size, encoder_conv_kernel_stride, encoded_h, device):
        super(WaveletEncoderDecoder, self).__init__()
        self.num_channels = num_channels

        self.seq_len = seq_len
        self.heads = heads
        self.encoded_h = encoded_h
        self.device = device
        self.conv_kernel_size = encoder_conv_kernel_size
        self.conv_kernel_stride = encoder_conv_kernel_stride

        L_out = self.seq_len + 2 * (self.conv_kernel_size//2) - 1 * (self.conv_kernel_size - 1) - 1
        L_out = floor(L_out / self.conv_kernel_stride) + 1

        self.window_embedder = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, 
            self.conv_kernel_size, stride=self.conv_kernel_stride,
            padding=self.conv_kernel_size//2),
            nn.BatchNorm1d(self.num_channels), # Same as Layer Norm
            nn.Dropout1d(0.1),
            nn.GELU(),
            nn.Linear(L_out, self.encoded_h)
        ).to(self.device)

        self.gnn_encoder = GATConv(encoded_h, self.encoded_h, heads=self.heads, concat=False).to(self.device)
        self.gnn_bnorm = nn.BatchNorm1d(self.num_channels).to(self.device)
        self.dropout = nn.Dropout1d(0.1)
        self.act = nn.GELU()
        self.lin = nn.Linear(self.encoded_h, self.encoded_h).to(self.device)

        self.learnable_padding = torch.nn.Parameter(torch.normal(0, self.encoded_h**(-0.5), size=(self.num_channels, self.seq_len - self.encoded_h), requires_grad=True)).to(self.device)
        self.transformer_decoder1 = nn.TransformerDecoderLayer(self.num_channels, nhead=1, dim_feedforward=L_out, activation=nn.GELU(), batch_first=True).to(self.device)
        self.transformer_decoder2 = nn.TransformerDecoderLayer(self.num_channels, nhead=1, dim_feedforward=L_out, activation=nn.GELU(), batch_first=True).to(self.device)

    def getEncoderParamCount(self):
        gnn_encoder_count = sum(p.numel() for p in self.gnn_encoder.parameters() if p.requires_grad)
        return gnn_encoder_count

    def getDecoderParamCount(self):
        transformer_decoder1_count = sum(p.numel() for p in self.transformer_decoder1.parameters() if p.requires_grad)
        transformer_decoder2_count = sum(p.numel() for p in self.transformer_decoder1.parameters() if p.requires_grad)
        return transformer_decoder1_count + transformer_decoder2_count

    def forward(self, graph, x):
        patch_embedding = self.act(self.window_embedder(x))
        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)
        gnn_encoder_input = patch_embedding.view(-1, patch_embedding.shape[-1])
        encoding = self.gnn_encoder(gnn_encoder_input, edge_index, edge_dist)
        encoding = encoding.view(-1, self.num_channels, self.encoded_h)
        encoding = self.gnn_bnorm(encoding)
        encoding = self.dropout(encoding)
        encoding = self.lin(encoding)

        decoding_tgt = torch.cat([encoding, self.learnable_padding.repeat((x.shape[0], 1, 1))], dim=-1).to(encoding.device)
        decoding = self.transformer_decoder1(tgt=decoding_tgt.permute([0, 2, 1]), memory=encoding.permute([0, 2, 1]))
        # Skip Connnection
        decoding = decoding.permute([0, 2, 1]) + decoding_tgt
        decoding_skip = self.transformer_decoder2(tgt=decoding.permute([0, 2, 1]), memory=encoding.permute([0, 2, 1]))
        # Skip Connnection
        decoding = decoding_skip.permute([0, 2, 1]) + decoding
        return encoding, decoding

Initialize Encoders for each wavelet band and 
put them into a single object.
class MENDRWindowEncoder(nn.Module):
    def __init__(self, device):
        super(MENDRWindowEncoder, self).__init__()
        self.device = device
        self.encoder_decoders = nn.ParameterDict({
            'delta': WaveletEncoderDecoder(
                    num_channels = 19,
                    seq_len = 246,
                    heads = 4,
                    encoded_h = 126,
                    encoder_conv_kernel_size = 2,
                    encoder_conv_kernel_stride = 2,
                    device = device
                    ),

            'theta': WaveletEncoderDecoder(
                    num_channels = 19,
                    seq_len = 246,
                    heads = 4,
                    encoded_h = 126,
                    encoder_conv_kernel_size = 2,
                    encoder_conv_kernel_stride = 2,
                    device = device
                    ),
            'alpha': WaveletEncoderDecoder(
                    num_channels = 19,
                    seq_len = 486,
                    heads = 4,
                    encoded_h = 246,
                    encoder_conv_kernel_size = 2,
                    encoder_conv_kernel_stride = 2,
                    device = device 
                    ),

            'beta': WaveletEncoderDecoder(
                    num_channels = 19,
                    seq_len = 966,
                    heads = 4,
                    encoded_h = 246,
                    encoder_conv_kernel_size = 2,
                    encoder_conv_kernel_stride = 2,
                    device = device
                    ),

            'gamma': WaveletEncoderDecoder(
                    num_channels = 19,
                    seq_len = 1925,
                    heads = 4,
                    encoded_h = 492,
                    encoder_conv_kernel_size = 4,
                    encoder_conv_kernel_stride = 4,
                    device = device
                    ),
        })

    def forward(self, graph, data):
        #assert data.keys() == self.encoder_decoders.keys()
        output = {}
        for band, band_decomposition in data.items():
            output[band] = self.encoder_decoders[band](graph, band_decomposition)
        return output

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if finetuning:
            self.mask_replacement.requires_grad = False

    def init_params(self, module):
         if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
'''