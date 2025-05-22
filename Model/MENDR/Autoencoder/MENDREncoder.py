import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor
from .MENDRReconstructionDecoder import MENDRReconstructionDecoder, SEBasicBlock
from ..MENDRCommon import Std_Norm

'''
We break each Wavelet sequence into patches. There are two patch scales: Super-Patch and Sub-Patch. 
By default super-patches represent 10 seconds and are what are first fed into the encoder.
The encoder then breaks the data up into smaller Sub-Patches (approx 1 sec by default).
Decoders are used for Super-Patch reconstruction to ensure embeddings have a real, physical
intepretation. 
'''
class WaveletEncoderDecoder(nn.Module):
    def __init__(self, num_channels, sub_patch_size, super_patch_seq_len,
                encoded_h, hidden_encoded_ratio, n_gnn_transformer_layers, num_subjects, device):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dropout = nn.Dropout1d(0.1)
        self.patch_size = sub_patch_size
        self.stride = self.patch_size // 2
        self.device = device
        self.encoded_h = encoded_h
        assert isinstance(hidden_encoded_ratio, int) and hidden_encoded_ratio > 0, "hidden_encoded_ratio must be a positive integer"
        self.hidden_encoded_ratio = hidden_encoded_ratio
        self.seq_len = super_patch_seq_len
        self.act = nn.GELU()

        self.gnn_spatial_harmonizer = GNNSpatialHarmonizer(num_channels=19, num_features=self.seq_len,
                                                           device=self.device, n_gnn_transformer_layers=n_gnn_transformer_layers,
                                                           heads=self.hidden_encoded_ratio)

        # Pre-Harmonization Patch Embedder 
        self.L_out_1 = self.seq_len + 2 * (0) - 1 * (self.patch_size - 1) - 1
        self.L_out_1 = floor((self.L_out_1 / self.stride) + 1)
        self.patch_embedder = nn.Conv1d(in_channels=19, out_channels=encoded_h, kernel_size=self.patch_size, stride=self.stride).to(self.device)
        self.patch_norm1 = nn.LayerNorm((encoded_h, self.L_out_1))
        self.hidden_encoded_h = self.hidden_encoded_ratio * self.L_out_1
        self.patch_embedder_lin = nn.Sequential(self.act, nn.Linear(self.L_out_1, self.hidden_encoded_h)).to(self.device)

        # Post-Harmonization Patch Embedder
        self.L_out_2 = self.hidden_encoded_h + 2 * (0) - 1 * (2 - 1) - 1
        self.L_out_2 = floor(self.L_out_2 / 1) + 1
        self.patch_embedder2 = nn.Conv1d(in_channels=encoded_h, out_channels=encoded_h, kernel_size=2, stride=1, groups=1, padding=0).to(self.device)
        self.patch_norm2 = nn.LayerNorm((encoded_h, self.L_out_2))
        self.hidden_encoded_length = self.L_out_2
        self.patch_embedder2_lin = nn.Sequential(self.act, nn.Linear(self.L_out_2, self.hidden_encoded_length)).to(self.device)
        self.SEBlock = SEBasicBlock(encoded_h, encoded_h, reduction=19).to(self.device)

        # Subject Embeddings
        self.subject_embeddings = nn.Embedding(num_embeddings=num_subjects,
                                    embedding_dim=self.hidden_encoded_length).to(self.device)
        # Decoders
        self.reconstruction_decoder = MENDRReconstructionDecoder(num_channels=19, sub_patch_size=self.patch_size,
                                                                 encoded_h=encoded_h, hidden_seq_length=self.hidden_encoded_length,
                                                                 seq_len=self.seq_len, device=self.device)

    def getEncoderParamCount(self):
        patch_embedder_count = sum(p.numel() for p in self.patch_embedder.parameters() if p.requires_grad)
        patch_norm1_count = sum(p.numel() for p in self.patch_norm1.parameters() if p.requires_grad)
        patch_embedder_lin_count = sum(p.numel() for p in self.patch_embedder_lin.parameters() if p.requires_grad)
        patch_embedder2_count = sum(p.numel() for p in self.patch_embedder2.parameters() if p.requires_grad)
        patch_norm2_count = sum(p.numel() for p in self.patch_norm2.parameters() if p.requires_grad)
        patch_embedder2_lin_count = sum(p.numel() for p in self.patch_embedder2_lin.parameters() if p.requires_grad)
        gnn_spatial_harmonizer_count = sum(p.numel() for p in self.gnn_spatial_harmonizer.parameters() if p.requires_grad)
        return patch_embedder_count + patch_norm1_count + patch_embedder_lin_count + patch_embedder2_count + patch_norm2_count + patch_embedder2_lin_count + gnn_spatial_harmonizer_count

    def disableDecoder(self):
        self.subject_embeddings = None
        self.reconstruction_decoder = None

    def getDecoderParamCount(self):
        reconstruction_decoder_count = sum(p.numel() for p in self.reconstruction_decoder.parameters() if p.requires_grad)
        return reconstruction_decoder_count

    def getEncoderParams(self):
        patch_embedder_params = list(self.patch_embedder.parameters())
        patch_norm1_params = list(self.patch_norm1.parameters())
        patch_embedder_lin_params = list(self.patch_embedder_lin.parameters())
        patch_embedder2_params = list(self.patch_embedder2.parameters())
        patch_norm2_params = list(self.patch_norm2.parameters())
        patch_embedder2_lin_params = list(self.patch_embedder2_lin.parameters())
        gnn_encoder_params = list(self.gnn_spatial_harmonizer.parameters())
        return patch_embedder_params + patch_norm1_params + patch_embedder_lin_params + patch_embedder2_params + patch_norm2_params + patch_embedder2_lin_params + gnn_encoder_params
                
    def getDecoderParams(self):
        return list(self.reconstruction_decoder.parameters())

    def forward(self, graph, x, subjects):
        # x: [Batch Size, Patches, Channels, Time Steps]
        B, P, C, T = x.shape
        x = x.reshape(B*P, C, T)
        # x: [Batch Size * Patches, Channels, Time Steps]
        x = self.channel_dropout(x)
        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)
        x = self.gnn_spatial_harmonizer(x, edge_index, edge_dist, B, P, C)

        x = self.patch_embedder(x)
        x = self.SEBlock(x)
        x = self.patch_norm1(x)
        x = self.patch_embedder_lin(x)
        # x: [Batch Size * Patches, Channels, self.L_out_1]

        x = self.patch_embedder2(x)
        x = self.patch_norm2(x)
        x = self.patch_embedder2_lin(x)

        # x: [Batch Size * Patches, self.encoded_h, self.L_out_2]
        x = x.reshape(B, P, self.encoded_h, self.hidden_encoded_length)
        decoding = None
        if self.reconstruction_decoder is not None and self.subject_embeddings is not None:
            decoding = x.clone()
            subject_embeddings = self.subject_embeddings(subjects.long()).unsqueeze(1).unsqueeze(1) # Make it [Batch, 1, 1, self.hidden_encoded_length]
            decoding =  decoding + subject_embeddings
            # decoding: [Batch Size * Patches, Channels, self.seq_len]
            decoding = decoding.reshape(B * P, self.encoded_h, self.hidden_encoded_length)
            decoding = self.reconstruction_decoder(decoding)
        return x, decoding

class GNNSpatialHarmonizer(nn.Module):
    def __init__(self, num_channels, num_features, device, n_gnn_transformer_layers=2, heads=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.n_gnn_transformer_layers = n_gnn_transformer_layers
        self.gnn_transformers = nn.ModuleList([GNNTransformer(num_features=self.num_features, num_channels=num_channels, 
                                               device=device, hidden_ratio=4, heads=heads) for _ in range(self.n_gnn_transformer_layers)])

    def forward(self, x, edge_index, edge_dist, B, P, C):
        # x: [Batch Size*Patches, Channels, num_features]
        for gnn_transformer in self.gnn_transformers:
            x = gnn_transformer(x, edge_index, edge_dist, B, P, C)
        return x

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
            channel_encoding = self.gnn_channel_encoder(gnn_channel_encoder_input, edge_index, edge_dist)
            # channel_encoding: [Batch Size, Channels, Time Steps, self.L_out_1, self.num_features]
            x[:, patch_idx, :, :] = x[:, patch_idx, :, :] + channel_encoding.reshape(B, C, self.num_features)

        x = x.view(B*P, C, self.num_features)
        # x: [Batch Size * Patches, Channels, self.hidden_ratio * self.num_features]
        x = self.layer_norm1(x)
        x = x + self.gnn_lin(x)
        x = self.layer_norm2(x)
        return x

'''
Initialize Encoders for each wavelet band and 
put them into a single object.
'''
class MENDRPatchEncoder(nn.Module):
    def __init__(self, 
                num_channels, 
                sampling_rate,
                super_patch_seconds,
                hop_length,
                delta_encoded_h,
                theta_encoded_h,
                alpha_encoded_h,
                beta_encoded_h,
                gamma_encoded_h,
                hidden_encoded_ratio,
                n_gnn_transformer_layers,
                num_subjects,
                device,
                high_encoded_h=None,
                ):
        super().__init__()
        self.device = device
        self.sampling_rate = sampling_rate
        self.super_patch_seconds = super_patch_seconds
        self.hop_length = hop_length

        
        # Each represents one second of data
        self.SUPPORTED_WAVELET_LENGTHS = {
            128 : {
                'delta': 4, # 0-4Hz, 1/32 
                'theta': 4, # 4-8Hz, 1/32
                'alpha': 8, # 8-16Hz, 1/16
                'beta': 16, # 16-32Hz, 1/8
                'gamma': 32, # 32-64Hz, 1/4
                'high': 64, # 64-128Hz, 1/2
            }, 
            256 : { 
                #TODO
            }
        }

        self.WAVELET_LENGTHS = self.SUPPORTED_WAVELET_LENGTHS[self.sampling_rate]
        self.WAVELET_SUPER_PATCH_LENGTHS = dict()
        self.WAVELET_SUPER_PATCH_HOP_LENGTHS = dict()
        for band, second_length in self.WAVELET_LENGTHS.items():
            self.WAVELET_SUPER_PATCH_LENGTHS[band] = self.super_patch_seconds * second_length
            self.WAVELET_SUPER_PATCH_HOP_LENGTHS[band] = int(self.WAVELET_SUPER_PATCH_LENGTHS[band] * self.hop_length)

        self.patch_normalizers = {
            'delta': Std_Norm(),
            'theta': Std_Norm(),
            'alpha': Std_Norm(),
            'beta': Std_Norm(),
            'gamma': Std_Norm(),
        }

        self.encoder_decoders = {
            'delta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['delta'],
                    encoded_h=delta_encoded_h,
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['delta'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'theta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['theta'],
                    encoded_h=theta_encoded_h,
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['theta'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'alpha': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['alpha'],
                    encoded_h=alpha_encoded_h,
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['alpha'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    num_subjects=num_subjects,
                    device = device 
                    ),
            'beta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['beta'],
                    encoded_h=beta_encoded_h,
                    super_patch_seq_len = self.WAVELET_SUPER_PATCH_LENGTHS['beta'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'gamma': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size = self.WAVELET_LENGTHS['gamma'],
                    encoded_h = gamma_encoded_h,
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['gamma'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    num_subjects=num_subjects,
                    device = device
                    ),
        }

        if high_encoded_h != None:
            self.encoder_decoders['high'] = WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size = self.WAVELET_LENGTHS['high'],
                    encoded_h = high_encoded_h,
                    super_patch_seq_len = self.WAVELET_SUPER_PATCH_LENGTHS['high'],
                    hidden_encoded_ratio=hidden_encoded_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    device = device
                    )
            self.patch_normalizers['high'] = LayerNormChannelOnly(num_channels=num_channels)
        self.encoder_decoders = nn.ParameterDict(self.encoder_decoders)
        self.patch_normalizers = nn.ParameterDict(self.patch_normalizers)

        for band, encoder_decoder in self.encoder_decoders.items():
            print(f"{band} Hidden Token Sequence Length: {encoder_decoder.hidden_encoded_length}")

    def forward(self, data):
        patchified_data = self._super_patchify(data)
        graphs = data['graph']
        try: # used only when decoder frozen
            subjects = data['subject_idx']
        except KeyError:
            subjects = None
        assert patchified_data.keys() == self.encoder_decoders.keys()
        encodings = {}
        decodings = {}
        for band, band_decomposition in patchified_data.items():
            encoding, decoding = self.encoder_decoders[band](graphs, band_decomposition, subjects)
            encodings[band] = encoding
            decodings[band] = decoding
        return patchified_data, encodings, decodings
    
    def _super_patchify(self, data):
        patchified_data = dict()
        min_patches = None # Sometimes the number of patches is different for each band
        for band, second_length in self.WAVELET_LENGTHS.items():
            if band in data: # Sometimes we exclude HIGH
                wavelet_data = data[band] # [Batches, Channels, Time Steps]
                patched_wavelet_data = []
                wavelet_super_patch_length = self.WAVELET_SUPER_PATCH_LENGTHS[band]
                wavelet_super_patch_hop_length = self.WAVELET_SUPER_PATCH_HOP_LENGTHS[band]
                for i in range(0, wavelet_data.shape[-1], wavelet_super_patch_hop_length):
                    if i + wavelet_super_patch_length <= wavelet_data.shape[-1]:
                        patched_wavelet_data.append(wavelet_data[:, :, i:i + wavelet_super_patch_length])
                patched_wavelet_data = torch.stack(patched_wavelet_data, dim=1) # [Batch, Patches, C, T]
                if min_patches is None:
                    min_patches = patched_wavelet_data.shape[1]
                else:
                    min_patches = min(min_patches, patched_wavelet_data.shape[1])
                patchified_data[band] = patched_wavelet_data

        # Truncate patches to the minimum number of patches
        for band, band_decomposition in patchified_data.items():
            patchified_data[band] = band_decomposition[:, :min_patches]
            patchified_data[band] = self.patch_normalizers[band](patchified_data[band])
        return patchified_data

    # For downstream tasks
    def reset_subject_embedding(self, num_subjects):
        self.subject = nn.Embedding(num_embeddings=num_subjects,
                                    embedding_dim=1)

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