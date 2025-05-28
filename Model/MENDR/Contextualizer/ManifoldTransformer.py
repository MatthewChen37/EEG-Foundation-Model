import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes, LogEuclidLayerNorm

class ManifoldTransformer(nn.Module):
    def __init__(self, device, encoded_h, hidden_scale=1.5, norm_output=True):
        super().__init__()
        self.encoded_h = encoded_h
        self.hidden_scale = hidden_scale
        self.device = device
        self.norm_output = norm_output

        self.manifold_self_attention = AttentionManifold(self.encoded_h, self.encoded_h, self.device)
        self.activation = SPDRectified()
        self.manifold_self_spd_transform = nn.Sequential(SPDTransform(self.encoded_h, int(self.hidden_scale * self.encoded_h), self.device),
                                                         self.activation,
                                                         SPDTransform(int(self.hidden_scale * self.encoded_h), self.encoded_h, self.device))

        self.trace_normalization = BatchTraceNormalization(self.device)
        self.layer_normalization = LogEuclidLayerNorm(self.device, self.encoded_h)

    def forward(self, x, batch_size, num_patches):
        # X is list of [Batch_Size, epochs, C, C]

        x_res, shape = self.manifold_self_attention(x)
        # x_res is [Batch_Size*epochs, C, C]
        x = x.view(x_res.shape) + x_res # Add and norm
        x = self.layer_normalization(x)
        x = x + self.manifold_self_spd_transform(x)
        if self.norm_output:
            x = self.layer_normalization(x)
        x = x.view(batch_size, num_patches, self.encoded_h, self.encoded_h)

        # Symmetrize Due to Numeric Instability
        x = 0.5 * (x + x.transpose(2, 3))
        return x












