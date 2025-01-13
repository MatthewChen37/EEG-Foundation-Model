import copy
import torch
import numpy as np
from torch import nn
from .layers import Permute, Flatten

'''
Based on:
1. https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py
2. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/layers.py
3. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/utils.py
'''
class Contextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # Initialize replacement vector with 0's
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)),
                                                   requires_grad=True)

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x = x.clone()
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x = x.clone()
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)
        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def _make_mask(shape, p, total, span, allow_no_inds=False):
    # Note that shape is (batch_size, seq_len) and total = seq_len.
    # We do not care about the features because all features are masked, we only care about the time points.
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    # Iterate through each item in the batch.
    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            # For each time point, generate a random number and if it is less than p, add it to the mask seeds.
            # There is an index by 0 ([0]) because np.nonzero returns a tuple of arrays.
            # Mask seeds are the start indicies of the span.
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]
        spans = _make_span_from_seeds(mask_seeds, span, total=total)
        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            # Break when i is greater than or equal to the number of time points.
            # Remember that the total is the number of time points and 
            # the range of the loop could be greater than the total.
            # This means that a mask is not always equal to span and could be less.
            if total is not None and i >= total:
                break
            elif i not in inds:
                # Add the index to the list of indices.
                inds.append(int(i))
    return np.array(inds)


if __name__ == "__main__":
    contextualizer = Contextualizer(512, layer_drop=0.01)
    
    mask = _make_mask((32, 512), 0.1, 512, 6)
    print(mask)
    print("Mask Shape:", mask.shape, "Mask Rate:", mask.float().mean().item())
    print("Mask sum:", mask.sum())
    print("Contextualizer Output Shape:",
           contextualizer(torch.randn(32, 512, 512), mask_t=mask).shape)