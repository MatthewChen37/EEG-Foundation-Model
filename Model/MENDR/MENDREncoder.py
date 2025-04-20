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
    def __init__(self, num_channels, sub_patch_size, super_patch_seq_len, encoded_h, device):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dropout = nn.Dropout1d(0.2)
        self.patch_size = sub_patch_size
        self.stride = self.patch_size // 2
        self.device = device
        self.encoded_h = encoded_h
        self.seq_len = super_patch_seq_len

        self.act = nn.GELU()
        self.L_out_1 = self.seq_len + 2 * (0) - 1 * ((self.patch_size) - 1) - 1
        self.L_out_1 = floor(self.L_out_1 / (self.stride)) + 1
        self.patch_embedder = nn.Conv1d(in_channels=19, out_channels=19, kernel_size=self.patch_size, stride=self.stride, groups=1).to(self.device)
        self.patch_embedder_lin = nn.Sequential(self.act, nn.Linear(self.L_out_1, 2 * self.L_out_1)).to(self.device)

        #self.gnn_channel_encoder = SplineConv(L_out, self.seq_len, dim=1, kernel_size=3).to(self.device)
        self.gnn_channel_encoder = GATConv(2 * self.L_out_1, 2 * self.L_out_1, heads=2, concat=False).to(self.device)
        self.gnn_lin1 = nn.Sequential(self.act, nn.Linear(2 * self.L_out_1, 4 * self.L_out_1), self.act, nn.Linear(4 * self.L_out_1, 2 * self.L_out_1)).to(self.device)

        self.layer_norm1 = nn.LayerNorm((19, 2 * self.L_out_1)).to(self.device)
        self.layer_norm2 = nn.LayerNorm((19, 2 * self.L_out_1)).to(self.device)
        self.layer_norm3 = nn.LayerNorm((19, 2 * self.L_out_1)).to(self.device)

        self.L_out_2 = 2 * self.L_out_1 + 2 * (0) - 1 * (2 - 1) - 1
        self.L_out_2 = floor(self.L_out_2 / 1) + 1
        self.patch_embedder2 = nn.Conv1d(in_channels=19, out_channels=encoded_h, kernel_size=2, stride=1, groups=1, padding=0).to(self.device)
        self.patch_embedder2_lin = nn.Sequential(self.act, nn.Linear(self.L_out_2, self.L_out_2)).to(self.device)

        # Decoders
        self.decode_L_out = (self.L_out_2 - 1) * self.stride - 2 * 0 + 1 * (self.patch_size - 1) + 0 + 1
        self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels=encoded_h, out_channels=encoded_h * 2, kernel_size=self.patch_size, stride=self.stride, groups=1).to(self.device), self.act)
        self.decode_L_out = (self.decode_L_out - 1) * (self.stride) - 2 * 0 + 1 * (2 - 1) + 0 + 1
        self.up2 = nn.Sequential(nn.ConvTranspose1d(in_channels=encoded_h * 2, out_channels=encoded_h * 2, kernel_size=2, stride=self.stride, groups=1).to(self.device), self.act)
        self.decode_L_out = self.decode_L_out + 2 * 0 - 1 * (encoded_h - 1) - 1
        self.decode_L_out = floor((self.decode_L_out / (self.stride)) + 1)
        self.up3 = nn.Sequential(nn.Conv1d(in_channels=encoded_h * 2, out_channels=19, kernel_size=encoded_h, stride=self.stride, groups=1).to(self.device), self.act)
        self.up4 = nn.Linear(self.decode_L_out, self.seq_len).to(self.device)
               
    def getEncoderParamCount(self):
        patch_embedder_count = sum(p.numel() for p in self.patch_embedder.parameters() if p.requires_grad)
        patch_embedder_lin_count = sum(p.numel() for p in self.patch_embedder_lin.parameters() if p.requires_grad)
        patch_embedder2_lin_count = sum(p.numel() for p in self.patch_embedder2_lin.parameters() if p.requires_grad)
        gnn_encoder_count = sum(p.numel() for p in self.gnn_channel_encoder.parameters() if p.requires_grad)
        gnn_lin = sum(p.numel() for p in self.gnn_lin1.parameters() if p.requires_grad)

        return patch_embedder_count + gnn_encoder_count + patch_embedder_lin_count + patch_embedder2_lin_count + gnn_lin

    def getDecoderParamCount(self):
        up1_count = sum(p.numel() for p in self.up1.parameters() if p.requires_grad)
        up2_count = sum(p.numel() for p in self.up2.parameters() if p.requires_grad)
        up3_count = sum(p.numel() for p in self.up3.parameters() if p.requires_grad)
        up4_count = sum(p.numel() for p in self.up4.parameters() if p.requires_grad)
        return up1_count + up2_count + up3_count + up4_count

    def getDecoderParams(self):
        return list(self.up1.parameters()) + list(self.up2.parameters()) + list(self.up3.parameters()) + list(self.up4.parameters())

    def forward(self, graph, x):
        # x: [Batch Size, Patches, Channels, Time Steps]
        B, P, C, T = x.shape
        x = x.view(B*P, C, T)
        # x: [Batch Size * Patches, Channels, Time Steps]
        x = self.channel_dropout(x)
        x = self.patch_embedder(x)
        x = self.patch_embedder_lin(x)
        x = self.layer_norm1(x)
        # x: [Batch Size * Patches, Channels, self.L_out_1]

        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)

        x = x.view(B, P, C, 2*self.L_out_1)
        # x: [Batch Size, Patches, Channels, self.L_out_1]
        for patch_idx in range(x.shape[1]):
            gnn_channel_encoder_input = x[:, patch_idx, :, :].reshape(B * C, 2 * self.L_out_1).clone() # TODO: without this clone I get an inplace modification error, why?
            # gnn_channel_encoder_input: [Batch Size * Channels (Each entry is a node), self.L_out_1]
            channel_encoding = self.gnn_channel_encoder(gnn_channel_encoder_input, edge_index, edge_dist)
            # channel_encoding: [Batch Size, Channels, Time Steps, self.L_out_1]
            x[:, patch_idx, :, :] = x[:, patch_idx, :, :] + channel_encoding.reshape(B, C, 2 * self.L_out_1)

        x = x.view(B*P, C, 2 * self.L_out_1)
        # x: [Batch Size * Patches, Channels, self.L_out_1]

        x = self.layer_norm2(x)
        x = x + self.gnn_lin1(x)
        x = self.layer_norm3(x)

        x = self.patch_embedder2(x)
        x = x + self.patch_embedder2_lin(x)
        # x: [Batch Size * Patches, self.encoded_h, self.L_out_2]
        decoding = x.clone().detach() # For torchjd, create separate autograd graph for decoding so that task parameters are not included in shared parameters
        decoding.requires_grad = True # Just to be safe
        #decoding = x.clone() 
        x = x.reshape(B, P, self.encoded_h, self.L_out_2)
        # x: [Batch Size, Patches, self.encoded_h, self.L_out_2]
        #decoding = self.up1(x)
        #print(decoding.shape)
        #print(decoding.shape)
        decoding = self.up1(decoding)
        #print(decoding.shape)
        decoding = self.channel_dropout(decoding)
        decoding = self.up2(decoding)
        #print(decoding.shape)
        decoding = self.up3(decoding)
        #print(decoding.shape)
        decoding = self.up4(decoding)
        # decoding: [Batch Size * Patches, Channels, self.seq_len]

        return x, decoding

'''
Initialize Encoders for each wavelet band and 
put them into a single object.
'''
class MENDRPatchEncoder(nn.Module):
    def __init__(self, 
                num_channels, 
                delta_sub_patch_size,
                theta_sub_patch_size,
                alpha_sub_patch_size,
                beta_sub_patch_size,
                gamma_sub_patch_size,
                high_sub_patch_size,
                delta_encoded_h,
                theta_encoded_h,
                alpha_encoded_h,
                beta_encoded_h,
                gamma_encoded_h,
                high_encoded_h,
                delta_super_patch_seq_len,
                theta_super_patch_seq_len,
                alpha_super_patch_seq_len,
                beta_super_patch_seq_len,
                gamma_super_patch_seq_len,
                high_super_patch_seq_len,
                device):
        super().__init__()
        self.device = device
        self.encoder_decoders = nn.ParameterDict({
            'delta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=delta_sub_patch_size,
                    encoded_h=delta_encoded_h,
                    super_patch_seq_len=delta_super_patch_seq_len,
                    device = device
                    ),
            'theta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=theta_sub_patch_size,
                    encoded_h=theta_encoded_h,
                    super_patch_seq_len=theta_super_patch_seq_len,
                    device = device
                    ),
            'alpha': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=alpha_sub_patch_size,
                    encoded_h=alpha_encoded_h,
                    super_patch_seq_len=alpha_super_patch_seq_len,
                    device = device 
                    ),
            'beta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=beta_sub_patch_size,
                    encoded_h=beta_encoded_h,
                    super_patch_seq_len = beta_super_patch_seq_len,
                    device = device
                    ),
            'gamma': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size = gamma_sub_patch_size,
                    encoded_h = gamma_encoded_h,
                    super_patch_seq_len=gamma_super_patch_seq_len,
                    device = device
                    ),
        })

    def forward(self, graph, data):
        assert data.keys() == self.encoder_decoders.keys()
        encodings = {}
        decodings = {}
        for band, band_decomposition in data.items():
            encoding, decoding = self.encoder_decoders[band](graph, band_decomposition)
            encodings[band] = encoding
            decodings[band] = decoding
        return encodings, decodings

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if finetuning:
            self.mask_replacement.requires_grad = False




# DEPRECATED
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