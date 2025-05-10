import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from .MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes

class ManifoldTransformer(nn.Module):
    def __init__(self, device, encoded_h, hidden_scale=1.5, patch_len=18):
        super().__init__()
        self.encoded_h = encoded_h
        self.hidden_scale = hidden_scale

        self.manifold_self_attention = AttentionManifold(self.encoded_h, self.encoded_h, self.device)
        self.activation = SPDRectified()
        self.manifold_self_spd_transform = nn.Sequential(SPDTransform(self.encoded_h, int(self.hidden_scale * self.encoded_h), self.device),
                                                         self.activation,
                                                         SPDTransform(int(self.hidden_scale * self.encoded_h), self.encoded_h, self.device))

        self.trace_normalization = BatchTraceNormalization(self.device)

    def forward(self, x, batch_size, num_patches, norm_output=True):
        # X is list of [Batch_Size, epochs, C, C]

        x_res, shape = self.manifold_self_attention(x)
        x = x + x_res.view(x.shape)

        epoched_shape = shape
        og_output_shape = x.shape
        x = x.view(batch_size, num_patches, self.encoded_h, self.encoded_h)
        x = self.trace_normalization(x)

        x = x + self.manifold_self_spd_transform(x)
        if norm_output:
            x = self.trace_normalization(x)

        return x, epoched_shape












