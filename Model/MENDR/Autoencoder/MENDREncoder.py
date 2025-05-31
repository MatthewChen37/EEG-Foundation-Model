import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import Sequential
from torch_geometric.data import Data, Batch
from math import floor
from einops import rearrange
from .MENDRReconstructionDecoder import MENDRReconstructionDecoder, SEBasicBlock
from ..MENDRCommon import std_norm
from Model.MENDR.Autoencoder.GNNSpatialHarmonizer import GNNSpatialHarmonizer, Dropout1dWithIndexTracking

'''
We break each Wavelet sequence into patches. There are two patch scales: Super-Patch and Sub-Patch. 
By default super-patches represent 10 seconds and are what are first fed into the encoder.
The encoder then breaks the data up into smaller Sub-Patches (approx 1 sec by default).
Decoders are used for Super-Patch reconstruction to ensure embeddings have a real, physical
intepretation. 
'''
class WaveletEncoderDecoder(nn.Module):
    def __init__(self, num_channels, sub_patch_size, super_patch_seq_len,
                hidden_gnn_mlp_ratio, n_gnn_transformer_layers, n_gnn_heads, num_subjects, device):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dropout = Dropout1dWithIndexTracking(p=0.1)
        self.patch_size = sub_patch_size
        self.device = device
        assert isinstance(hidden_gnn_mlp_ratio, int) and hidden_gnn_mlp_ratio > 0, "hidden_gnn_mlp_ratio must be a positive integer"
        self.hidden_gnn_mlp_ratio = hidden_gnn_mlp_ratio
        self.n_gnn_heads = n_gnn_heads
        self.seq_len = super_patch_seq_len
        self.act = nn.GELU()

        self.gnn_spatial_harmonizer = GNNSpatialHarmonizer(num_channels=self.num_channels, num_features=self.seq_len,
                                                        n_gnn_transformer_layers=n_gnn_transformer_layers,
                                                        hidden_ratio=hidden_gnn_mlp_ratio, heads=self.n_gnn_heads)

        self.SEBlock = SEBasicBlock(self.num_channels, self.num_channels, reduction=1).to(self.device)

        self.patch_embedder = PatchEmbedder(patch_size=self.patch_size, in_dim=1, out_dim=24, seq_len=self.seq_len).to(self.device)

        # Subject Embeddings
        self.subject_embeddings = nn.Embedding(num_embeddings=num_subjects,
                                    embedding_dim=self.seq_len).to(self.device)
        # Decoders
        self.reconstruction_decoder = MENDRReconstructionDecoder(num_channels=19, sub_patch_size=self.patch_size,
                                                                 encoded_h=24, hidden_seq_length=self.seq_len,
                                                                 seq_len=self.seq_len, device=self.device)

    def getEncoderParamCount(self):
        patch_embedder_count = sum(p.numel() for p in self.patch_embedder.parameters() if p.requires_grad)
        gnn_spatial_harmonizer_count = sum(p.numel() for p in self.gnn_spatial_harmonizer.parameters() if p.requires_grad)
        return patch_embedder_count + gnn_spatial_harmonizer_count

    def disableDecoder(self):
        self.subject_embeddings = None
        self.reconstruction_decoder = None

    def getDecoderParamCount(self):
        reconstruction_decoder_count = sum(p.numel() for p in self.reconstruction_decoder.parameters() if p.requires_grad)
        return reconstruction_decoder_count

    def getEncoderParams(self):
        patch_embedder_params = list(self.patch_embedder.parameters())
        gnn_encoder_params = list(self.gnn_spatial_harmonizer.parameters())
        return patch_embedder_params + gnn_encoder_params
                
    def getDecoderParams(self):
        return list(self.reconstruction_decoder.parameters())

    def forward(self, graph, x, subjects):
        # x: [Batch Size, Patches, Channels, Time Steps]
        B, P, C, T = x.shape
        x = self.channel_dropout(x)
        #print(B, P, C, T, x.shape)
        edge_index = graph.edge_index.to(self.device)
        edge_dist = graph.edge_attr.to(self.device)
        x = self.gnn_spatial_harmonizer(x, edge_index, edge_dist, B, P, C)
        # x: [Batch Size * Patches, Channels, Time Steps]
        x = self.SEBlock(x)
        x = rearrange(x, '(B P) C T -> B P C T', B=B, P=P, C=C, T=T)
        # x: [Batch Size, Patches, Channels, Time Steps]
        x = self.patch_embedder(x)
        # x: [Batch Size, Patches, Channels * patch_embedder.out_dim, Time Steps]

        decoding = None
        if self.reconstruction_decoder is not None and self.subject_embeddings is not None:
            decoding = x.clone()
            subject_embeddings = self.subject_embeddings(subjects.long()).unsqueeze(1).unsqueeze(1) # Make it [Batch, 1, 1, Time Steps]
            decoding = decoding + subject_embeddings
            # decoding: [Batch Size * Patches, Channels, self.seq_len]
            decoding = self.reconstruction_decoder(decoding, B, P, C, T)
        return x, decoding


'''
Initialize Encoders for each wavelet band and 
put them into a single object.
'''
class MENDRPatchEncoder(nn.Module):
    def __init__(self, 
                num_channels, 
                sampling_rate,
                super_patch_seconds,
                hidden_gnn_mlp_ratio,
                n_gnn_transformer_layers,
                n_gnn_heads,
                num_subjects,
                device,
                high_encoded_h=None,
                ):
        super().__init__()
        self.device = device
        self.sampling_rate = sampling_rate
        self.super_patch_seconds = super_patch_seconds

        
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
        for band, second_length in self.WAVELET_LENGTHS.items():
            self.WAVELET_SUPER_PATCH_LENGTHS[band] = self.super_patch_seconds * second_length

        self.patch_normalizer = std_norm()

        self.encoder_decoders = {
            'delta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['delta'],
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['delta'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'theta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['theta'],
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['theta'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'alpha': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['alpha'],
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['alpha'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    num_subjects=num_subjects,
                    device = device 
                    ),
            'beta': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size=self.WAVELET_LENGTHS['beta'],
                    super_patch_seq_len = self.WAVELET_SUPER_PATCH_LENGTHS['beta'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    num_subjects=num_subjects,
                    device = device
                    ),
            'gamma': WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size = self.WAVELET_LENGTHS['gamma'],
                    super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['gamma'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    num_subjects=num_subjects,
                    device = device
                    ),
        }

        if high_encoded_h != None:
            self.encoder_decoders['high'] = WaveletEncoderDecoder(
                    num_channels = num_channels,
                    sub_patch_size = self.WAVELET_LENGTHS['high'],
                    super_patch_seq_len = self.WAVELET_SUPER_PATCH_LENGTHS['high'],
                    hidden_gnn_mlp_ratio=hidden_gnn_mlp_ratio,
                    n_gnn_transformer_layers=n_gnn_transformer_layers,
                    n_gnn_heads=n_gnn_heads,
                    device = device
                    )
            self.patch_normalizers['high'] = LayerNormChannelOnly(num_channels=num_channels)
        self.encoder_decoders = nn.ParameterDict(self.encoder_decoders)

        for band, encoder_decoder in self.encoder_decoders.items():
            print(f"{band} Hidden Token Sequence Length: {encoder_decoder.seq_len}")

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
    

    '''
    All data is currently preprocessed as Channels x Time Steps, which enables us to 
    easily change the patch size and stride during run-time. This is not the most computationally
    efficient way to do this, but it makes it easier for ablation studies. A future optimization 
    is to set the patch size in pre-training (see CBraMods preprocessing). 

    Thus, patches are calculated during run-time.
    '''
    def _super_patchify(self, data):
        patchified_data = dict()
        for band, second_length in self.WAVELET_LENGTHS.items():
            if band in data: # Sometimes we exclude HIGH
                wavelet_data = data[band] # [Batches, Channels, Time Steps]
                patched_wavelet_data = []
                wavelet_super_patch_length = self.WAVELET_SUPER_PATCH_LENGTHS[band]
                '''
                #this is usually slower
                for i in range(0, wavelet_data.shape[-1], wavelet_super_patch_length):
                    if i + wavelet_super_patch_length <= wavelet_data.shape[-1]:
                        patched_wavelet_data.append(wavelet_data[:, :, i:i + wavelet_super_patch_length])
                patched_wavelet_data = torch.stack(patched_wavelet_data, dim=1) # [Batch, Patches, C, T]
                '''
                patched_wavelet_data = rearrange(wavelet_data, 'b c (pn pl) -> b pn c pl', pl=wavelet_super_patch_length)
                patchified_data[band] = self.patch_normalizer(patched_wavelet_data)
                #patchified_data[band] = patched_wavelet_data
        # Truncate patches to the minimum number of patches
        return patchified_data

    # For downstream tasks
    def reset_subject_embedding(self, num_subjects):
        self.subject = nn.Embedding(num_embeddings=num_subjects,
                                    embedding_dim=1)

class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, seq_len, in_dim=1, out_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = (patch_size // 2) + 1
        self.stride = 1
        self.padding = max(1, (self.patch_size - 1) // 2)

        self.in_dim = in_dim
        self.out_dim = out_dim
        # https://github.com/935963004/LaBraM/blob/5f5ec3e702199ef0f16ee0bbaa8c2997cb77b786/modeling_pretrain.py#L28

        L_out = self.seq_len + 2 * (self.padding) - 1 * (self.patch_size - 1) - 1
        L_out = floor((L_out / self.stride) + 1)
        assert L_out == self.seq_len, f"Patch Sequence Length {L_out} does not match {self.seq_len}, patch_size: {self.patch_size}, stride: {self.stride}"

        # Maintain sequence length
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GroupNorm(num_groups=4, num_channels=self.out_dim),
            nn.GELU(),
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GroupNorm(num_groups=4, num_channels=self.out_dim),
            nn.GELU(),
        )

        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1, self.patch_size), stride=(1, self.stride), padding=(0, self.padding)),
            nn.GroupNorm(num_groups=4, num_channels=self.out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        B, P, C, T = x.shape
        x = rearrange(x, 'B P C T -> B (P C) T', B=B, T=T)
        x = x.unsqueeze(1)
        # x: [Batch Size, 1, Patches * Channels, Time Steps]
        x = self.proj1(x)
        # x: [Batch Size, out_dim, Patches * Channels, Time Steps]
        x = self.proj2(x)
        # x: [Batch Size, out_dim, Patches * Channels, Time Steps]
        x = self.proj3(x)
        x = rearrange(x, 'B O (P C) T -> B P (C O) T', B=B, P=P, C=C, T=T, O=self.out_dim)
        # x: [Batch Size, Patches, Channels * out_dim, Time Steps]
        return x

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