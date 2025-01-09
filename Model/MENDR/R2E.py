import torch
import torch.nn as nn
from mAtt.mAtt import AttentionManifold, SPDRectified, SPDTangentSpace

'''
Convert Riemannian Embeddings to Euclidean Embeddings (R2E)
From https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py

The reason why we split into two classes is becauses it 
makes extracting the Riemannian embeddings easier.
'''
class R2E(nn.Module):
	def __init__(self, config):
		'''
		Args:
			config: a dictionary containing the following keys:
				- epochs: number of epochs	
		'''
		super().__init__()
		self.epochs = config.epochs
		self.tangent = SPDTangentSpace(12)


	def forward(self, x, shape):
		'''
		Args:
			x: a tensor
			shape: a tuple of the original shape of x		
		'''
		x = self.tangent(x)
		x = x.view(shape[0], shape[1], -1)
		return x
	
	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False