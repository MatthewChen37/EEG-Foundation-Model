import torch
import torch.nn as nn
import math

# Based on BENDR's Convolutional Position Encoding Scheme
# Positional as in "Temporal"
class PositionalEncoding(nn.Module):
	def __init__(self, encoded_h, patch_len, dropout=0.1):
		super().__init__()
		self.encoded_h = encoded_h
		self.len = patch_len

		# Asymmetric Conditional Positional Encoding (ACPE) like CBraMod
		conv = nn.Conv2d(self.len, self.len, kernel_size=(3, self.encoded_h), padding=(1, self.encoded_h // 2))
		nn.init.normal_(conv.weight, mean=0, std=1)
		nn.init.constant_(conv.bias, 0)
		conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
		self.conv = nn.Sequential(conv, nn.GELU(), nn.Dropout(p=dropout))
		self.W_out = self.encoded_h + 2 * (self.encoded_h // 2) - 1 * (self.encoded_h - 1) - 1
		self.W_out = math.floor((self.W_out / 1) + 1)

		self.conv_adj = None
		if encoded_h != self.W_out: # Non centerable convolution, similar to how BENDR handles it
			print(f"Modifying positional encoder to be centerable by adding an additional convolution layer {self.W_out} -> {self.encoded_h}")
			self.conv_adj = nn.Conv2d(self.W_out, self.encoded_h, kernel_size=1)

	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape ``[Batch, #patch, encoded_h, #time_step]``
		"""
		x = x.permute(0, 3, 1, 2)
		# x is now [Batch, #time_step, #patch, encoded_h]
		positional_encoding = self.conv(x)
		if self.conv_adj:
			# Positional Encoding is now [Batch, encoded_h, #time_step, #patch]
			positional_encoding = positional_encoding.permute(0, 3, 1, 2)
			positional_encoding = self.conv_adj(positional_encoding)
			# Positional encoding is now [Batch, #time_step, #patch, encoded_h]
			positional_encoding = positional_encoding.permute(0, 2, 3, 1)
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


if __name__ == "__main__":
	# Test Positional Encoding
	pe = PositionalEncoding(38, 40)
	x = torch.randn(2, 11, 38, 40)
	x = pe(x)
	assert x.shape == torch.Size([2, 11, 38, 40]), f"Positional Encoding Test Failed: {x.shape}"
	print("Positional Encoding Test Passed")