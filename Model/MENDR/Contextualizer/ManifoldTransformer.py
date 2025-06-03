import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified, tensor_log, tensor_exp, log_euclidean_distance, LogEuclideanMean
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes, LogEuclidLayerNorm
from einops import rearrange

class ManifoldTransformer(nn.Module):
    def __init__(self, encoded_h, hidden_scale=1.5, norm_output=True):
        super().__init__()
        self.encoded_h = encoded_h
        self.hidden_scale = hidden_scale
        self.norm_output = norm_output

        self.manifold_self_attention = AttentionManifold(self.encoded_h, self.encoded_h)
        self.activation = SPDRectified()
        self.manifold_self_spd_transform = nn.Sequential(SPDTransform(self.encoded_h, int(self.hidden_scale * self.encoded_h)),
                                                         self.activation,
                                                         SPDTransform(int(self.hidden_scale * self.encoded_h), self.encoded_h))
        self.riemannian_residual = _RiemannianResidual()

        self.trace_normalization = BatchTraceNormalization()
        self.layer_normalization = LogEuclidLayerNorm(self.encoded_h)

    def forward(self, x, batch_size, num_patches):
        # X is list of [Batch_Size, epochs, C, C]
        x_res, shape = self.manifold_self_attention(x)
        # x_res is [Batch_Size*epochs, C, C]
        x = rearrange(x, 'B P C C -> (B P) C C', B=batch_size, P=num_patches, C=self.encoded_h)
        x = self.riemannian_residual(x, x_res)
        x = self.layer_normalization(x)
        x = x + self.manifold_self_spd_transform(x)
        if self.norm_output:
            x = self.layer_normalization(x)
        x = rearrange(x, '(B P) C C -> B P C C', B=batch_size, P=num_patches, C=self.encoded_h)

        # Symmetrize/Regularize Due to Numeric Instability
        x = 0.5 * (x + x.transpose(2, 3))
        return x

# https://proceedings.neurips.cc/paper_files/paper/2023/file/c868aa7437dc9b29e674cd2e25689021-Paper-Conference.pdf
# For some cool theory!!!$!!
class _RiemannianResidual(nn.Module):
    def __init__(self):
        super().__init__()

    # Identical to:
    # https://github.com/CUAI/Riemannian-Residual-Neural-Networks/blob/a3b4d6cfd066e636349311bc197ae06934cacf0c/rresnet/manifolds/spd.py#L100
    def forward(self, x, y, shape):
        # x: [Batch*#patches, C, C]
        # y: [Batch*#patches, C, C]
        x = tensor_exp(tensor_log(x) + tensor_log(y))
        return x