import torch
import torch.nn as nn
import math
import numpy as np

# Based on BENDR's Convolutional Position Encoding Scheme
# Positional as in "Temporal"
class PositionalEncoding(nn.Module):
	def __init__(self, device, encoded_h, patch_len, dropout=0.1):
		super().__init__()
		self.encoded_h = encoded_h
		self.len = patch_len
		self.device = device

		# Asymmetric Conditional Positional Encoding (ACPE) like CBraMod
		conv = nn.Conv2d(self.len, 1, kernel_size=(3, self.encoded_h), stride=(1, 1), padding=(1, (self.encoded_h - 1) // 2))
		nn.init.normal_(conv.weight, mean=0, std=1)
		nn.init.constant_(conv.bias, 0)
		conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
		#self.act = nn.GELU()
		self.conv = nn.Sequential(conv, nn.Dropout(p=dropout)).to(self.device)
		self.W_out = self.encoded_h + 2 * ((self.encoded_h - 1) // 2) - 1 * (self.encoded_h - 1) - 1
		self.W_out = math.floor((self.W_out / 1) + 1)
		#self.conv_adj = nn.Conv2d(self.W_out, self.encoded_h, kernel_size=(1, self.encoded_h), stride=(1, 1), padding=(0, 0)).to(self.device)

	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape ``[Batch, #patch, encoded_h, #time_step]``
		"""
		x = x.permute(0, 3, 1, 2)
		# x is now [Batch, #time_step, #patch, encoded_h]
		#print(x.shape, x, "Is Nan: ", torch.isnan(x).any())
		# Positional encoding is broadcast against x:
		# [Batch, #time_step, #patch, encoded_h] + [Batch, 1, #patch, encoded_h] = [Batch, #time_step, #patch, encoded_h]
		positional_encoding = self.conv(x)
		#positional_encoding = self.act(positional_encoding)
		# Positional Encoding is now [Batch, 1, #patch, W_out]
		# positional_encoding = self.conv_lin(positional_encoding)
		# Positional encoding is now [Batch, 1, #patch, encoded_h]
		x = x + positional_encoding 
		# x is now back to [Batch, #patch, encoded_h, #time_step]
		x = x.permute(0, 2, 3, 1)
		return x
	
class BatchTraceNormalization(nn.Module):
	def __init__(self, device, num_channels=19, epsilon=1e-5):
		super().__init__()
		self.num_channels = num_channels
		self.epsilon = epsilon
		self.device = device

	def forward(self, x):
		# Expects [B, C, C]
		trace = x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
		trace = trace.view(-1, 1, 1)
		trace = trace + self.epsilon*torch.ones(trace.shape).to(self.device)
		x /= trace
		identity = torch.eye(x.shape[-1], x.shape[-1], device=self.device).to(self.device).repeat(x.shape[0], 1, 1)
		x = x + (self.epsilon * identity)
		return x

def _make_mask_idxes(batch_size, num_patches, mask_ratio):
		num_masked = int(mask_ratio * num_patches)
		batch_mask_idxes = torch.zeros(batch_size, num_patches, dtype=torch.bool)
		for i in range(batch_size):
			indices = np.random.choice(num_patches, num_masked, replace=False)
			batch_mask_idxes[i,indices] = True
		return batch_mask_idxes

class LogEuclidLayerNorm(nn.Module):
    def __init__(self, device, num_channels=19, epsilon=1e-5):
        super().__init__()
        self.device       = device
        self.num_channels = num_channels
        self.epsilon      = epsilon
        # params
        self.gamma = nn.Parameter(torch.ones(num_channels, device=device))
        self.beta  = nn.Parameter(torch.zeros(num_channels, device=device))

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
        I = torch.eye(self.num_channels, device=self.device).unsqueeze(0)
        C_norm = C_norm + self.epsilon * I

        return C_norm

class Std_Norm(nn.Module):
	def __init__(self):
		super().__init__()
		#self.layer_norm = nn.LayerNorm(num_channels, elementwise_affine=False)

	def forward(self, x):
		# Expects x [Batch Size, Patches, Channels, Time Steps]
		# https://github.com/935963004/LaBraM/blob/5f5ec3e702199ef0f16ee0bbaa8c2997cb77b786/modeling_vqnsp.py#L143
		mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
		std = torch.std(x, dim=(1, 2, 3), keepdim=True)
		x = (x - mean) / std
		#x = self.layer_norm(x)
		return x

if __name__ == "__main__":
	# Test Positional Encoding
	pe = PositionalEncoding("cpu", 114, 37)
	print(f"Positional Encoding Test Parameters: {sum(p.numel() for p in pe.parameters() if p.requires_grad)}")
	x = torch.randn(4, 11, 114, 37)
	x = pe(x)
	assert x.shape == torch.Size([4, 11, 114, 37]), f"Positional Encoding Test Failed: {x.shape}"
	print("Positional Encoding Test Passed")