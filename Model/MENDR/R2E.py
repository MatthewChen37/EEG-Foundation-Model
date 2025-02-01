import torch
import torch.nn as nn
from .mAtt.mAtt import AttentionManifold, SPDRectified, SPDTangentSpace

'''
Convert Riemannian Embeddings to Euclidean Embeddings (R2E)
From https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py

The reason why we split into two classes is becauses it 
makes extracting the Riemannian embeddings easier.
'''
class R2E(nn.Module):
	def __init__(self, epochs):
		'''
		Args:
			config: a dictionary containing the following keys:
				- epochs: number of epochs	
		'''
		super().__init__()
		self.epochs = epochs
		self.tangent = SPDTangentSpace(32)
		self.layer_norm = nn.LayerNorm(528)
		self.ff1 = nn.Linear(528, 32)
		self.ff2 = nn.Linear(32, 32)
		self.gelu = nn.GELU()

	def forward(self, x, shape):
		'''
		Args:
			x: a tensor
			shape: a tuple of the original shape of x		
		'''
		x = self.tangent(x)
		x = x.view(shape[0], shape[1], -1)
		x = self.layer_norm(x)
		x = self.ff1(x)
		x = self.gelu(x)
		x = self.ff2(x)
		return x
	
	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False