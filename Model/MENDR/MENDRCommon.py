import torch
import torch.nn as nn
import math
import numpy as np
from einops import rearrange

# Based on CBraMod's Assymetric Conditional Positional Encoding (ACPE)
# Positional as in "Temporal" w.r.t to Patches
class PositionalEncoding(nn.Module):
	def __init__(self, num_channels, out_dim, patch_len):
		super().__init__()
		self.num_channels = num_channels
		self.out_dim = out_dim
		self.patch_len = patch_len

		# Asymmetric Conditional Positional Encoding (ACPE) like CBraMod
		self.conv = nn.Conv2d(in_channels=self.out_dim*self.patch_len, out_channels=self.out_dim*self.patch_len,
					kernel_size=(19, 3), stride=(1, 1), padding=(9, (3 - 1) // 2), groups=self.out_dim*self.patch_len)
		#self.act = nn.GELU()
		#self.W_out = self.encoded_h + 2 * ((self.encoded_h - 1) // 2) - 1 * (self.encoded_h - 1) - 1
		#self.W_out = math.floor((self.W_out / 1) + 1)
		#self.conv_adj = nn.Conv2d(self.W_out, self.encoded_h, kernel_size=(1, self.encoded_h), stride=(1, 1), padding=(0, 0)).to(self.device)

	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape [Batch Size, Patches, Channels, Time Steps* out_dim]
		"""
		B, P = x.shape[0], x.shape[1]
		x = rearrange(x, 'B P C (O T) -> B (O T) C P', B=B, P=P, C=self.num_channels, O=self.out_dim, T=self.patch_len)
		# x is now [Batch Size, out_dim * time_steps, Patches, Channels]
		positional_encoding = self.conv(x)
		# Positional Encoding is now [Batch, out_dim * time_steps, patches, encoded_h]
		x = x + positional_encoding 
		# x is now back to [Batch Size, Patches, Channels * out_dim, Time Steps]
		x = rearrange(x, 'B (O T) C P -> B P C (O T)', B=B, P=P, C=self.num_channels, O=self.out_dim, T=self.patch_len)
		return x

	
def _make_mask_idxes(batch_size, num_patches, mask_ratio):
		num_masked = int(mask_ratio * num_patches)
		batch_mask_idxes = torch.zeros(batch_size, num_patches, dtype=torch.bool)
		for i in range(batch_size):
			indices = np.random.choice(num_patches, num_masked, replace=False)
			batch_mask_idxes[i,indices] = True
		return batch_mask_idxes

class BatchTraceNormalization(nn.Module):
	def __init__(self, num_channels=19, epsilon=1e-5):
		super().__init__()
		self.num_channels = num_channels
		self.epsilon = epsilon

	def forward(self, x):
		# Expects [B, C, C]
		trace = x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
		trace = trace.view(-1, 1, 1)
		trace = trace + self.epsilon*torch.ones(trace.shape).to(x.device)
		x /= trace
		identity = torch.eye(x.shape[-1], x.shape[-1], device=x.device).repeat(x.shape[0], 1, 1)
		x = x + (self.epsilon * identity)
		return x

class LogEuclidLayerNorm(nn.Module):
    def __init__(self, num_channels=19, epsilon=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.epsilon      = epsilon
        # params
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta  = nn.Parameter(torch.zeros(num_channels))

    def forward(self, C):
        # assume SPD
        # Eigendecompose
		# D: [B,N], U: [B,N,N]
        D, U = torch.linalg.eigh(C)         

        # Log‐map + clamp
        D = D.clamp(min=self.epsilon)
        logD = torch.log(D)                    

        # LayerNorm in log‐space
		# [B,1]
        mu    = logD.mean(dim=-1, keepdim=True)
		# [B,1]      
        var   = logD.var(dim=-1, unbiased=False, keepdim=True)
		# [B,N]
        D_hat = (logD - mu) / torch.sqrt(var + self.epsilon)   

        # transform
		# [B,N]
        D_tilde = self.gamma * D_hat + self.beta     

        # exp, diagnolize, reconstruct
		# [B,N,N]
        C_norm = U @ torch.diag_embed(torch.exp(D_tilde)) @ U.transpose(-2, -1) 

        # ensure PD
        I = torch.eye(self.num_channels, device=C.device).unsqueeze(0)
        C_norm = C_norm + self.epsilon * I

        return C_norm

class std_norm(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		# Expects x [Batch Size, Patches, Channels, Time Steps]
		# To make each patch normalized
		# https://github.com/935963004/LaBraM/blob/5f5ec3e702199ef0f16ee0bbaa8c2997cb77b786/modeling_vqnsp.py#L143
		mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
		std = torch.std(x, dim=(1, 2, 3), keepdim=True)
		x = (x - mean) / std
		return x

if __name__ == "__main__":
	B, P, C, O, T = 4, 10, 19, 24, 64

	# Test Positional Encoding
	pe = PositionalEncoding(num_channels=C, out_dim=O, patch_len=T)
	print(f"Positional Encoding Test Parameters: {sum(p.numel() for p in pe.parameters() if p.requires_grad)}")
	x = torch.randn(B, P, C*O, T)
	x = pe(x)
	assert x.shape == torch.Size([B, P, C*O, T]), f"Positional Encoding Test Failed: {x.shape}"
	print("Positional Encoding Test Passed")