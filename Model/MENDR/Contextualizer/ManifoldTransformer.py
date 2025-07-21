import torch
import torch.nn as nn
import numpy as np
from Model.MENDR.mAtt.mAtt import E2R, AttentionManifold, SPDRectified, tensor_log, tensor_exp, log_euclidean_distance, LogEuclideanMean
from Model.MENDR.mAtt.spd import SPDTangentSpace, SPDTransform
from Model.MENDR.MENDRCommon import PositionalEncoding, BatchTraceNormalization, _make_mask_idxes, LogEuclidLayerNorm
from einops import rearrange

class ManifoldTransformer(nn.Module):
    def __init__(self, num_channels, norm_output=True):
        super().__init__()
        self.encoded_h = num_channels
        self.norm_output = norm_output

        #self.down_project = SPDTransform(self.encoded_h, self.encoded_h)
        self.manifold_self_attention = AttentionManifold(self.encoded_h, self.encoded_h)
        self.activation = SPDRectified()
        self.manifold_self_spd_transform = nn.Sequential(SPDTransform(self.encoded_h, int(1.5*self.encoded_h)),
                                                         self.activation,
                                                         SPDTransform(int(1.5*self.encoded_h), self.encoded_h))
        self.riemannian_residual = _RiemannianResidual()
        self.trace_normalization = BatchTraceNormalization()
        #self.layer_normalization = LogEuclidLayerNorm(self.encoded_h)

    def forward(self, x, batch_size, num_patches):
        # X is list of [Batch_Size, epochs, C, C]
        x_res, shape = self.manifold_self_attention(x)
        # x_res is [Batch_Size*epochs, C, C]
        x = rearrange(x, 'B P C1 C2 -> (B P) C1 C2', B=shape[0], P=shape[1], C1=self.encoded_h, C2=self.encoded_h)
        #x = self.down_project(x)
        x = self.riemannian_residual(x, x_res)
        x = self.trace_normalization(x)
        x_res = self.manifold_self_spd_transform(x)
        x = self.riemannian_residual(x, x_res)
        if self.norm_output:
            x = self.trace_normalization(x)
        x = rearrange(x, '(B P) C1 C2 -> B P C1 C2', B=shape[0], P=shape[1], C1=self.encoded_h, C2=self.encoded_h)
        # Symmetrize/Regularize Due to Numeric Instability
        # x = 0.5 * (x + x.transpose(-1, -2))
        return x

# https://proceedings.neurips.cc/paper_files/paper/2023/file/c868aa7437dc9b29e674cd2e25689021-Paper-Conference.pdf
# For some cool theory!!!$!!
class _RiemannianResidual(nn.Module):
    def __init__(self):
        super().__init__()

    # Identical to:
    # https://github.com/CUAI/Riemannian-Residual-Neural-Networks/blob/a3b4d6cfd066e636349311bc197ae06934cacf0c/rresnet/manifolds/spd.py#L100
    def forward(self, x, y):
        # x: [Batch*#patches, C, C]
        # y: [Batch*#patches, C, C]
        #x = 0.5 * (x + x.transpose(-1, -2))
        #y = 0.5 * (y + y.transpose(-1, -2))
        x = tensor_exp(tensor_log(x) + tensor_log(y))
        return x