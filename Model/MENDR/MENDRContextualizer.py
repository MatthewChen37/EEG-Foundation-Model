import torch
import torch.nn as nn
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
# TODO: Remove this import
import sys
sys.path.append("../")
from ..layers import Permute, Flatten

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class mATTContextualizer(nn.Module):
	def __init__(self, config):
		'''
		Args:
			config: a dictionary containing the following keys:
				- in_features: number of input features
				- dropout: dropout rate
				- epochs: number of epochs
				- position_encoder: position encoder (Typically 25 as in BENDR)
		'''
		super(mATTContextualizer, self).__init__()
		self.config = config
		self.in_features = config.in_features
		self._transformer_dim = config.in_features * 3
		self.dropout = config.dropout
		self.position_encoder = config.position_encoder > 0
		if self.position_encoder:
			self.relative_position = self._initializePositionEncoder(config)
		self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(config.in_features),
            nn.Dropout(config.dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(config.in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )
		# Initialize replacement vector with 0's
		self.mask_replacement = torch.nn.Parameter(torch.normal(0, self.in_features**(-0.5), size=(self.in_features,)),
                                                   requires_grad=True)
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		#E2R
		self.ract1 = E2R(config.epochs)
		#Riemannian Manifold Attention Module
		self.att = AttentionManifold(self._transformer_dim, self.in_features)
		self.ract2 = SPDRectified() 

	def forward(self, x, mask_t=None, mask_c=None):
		'''
		Args:
			x: a tensor of shape (batch_size, 1, channel, sample)
			mask_t: a tensor of shape (batch_size, 1, sample)
			mask_c: a tensor of shape (batch_size, 1, channel)
		Returns:
			x: a tensor of shape (batch_size, ???)
		'''
		if mask_t is not None:
			x = x.clone()
			x.transpose(2, 1)[mask_t] = self.mask_replacement
		if mask_c is not None:
			x = x.clone()
			x[mask_c] = 0
		if self.position_encoder:
			x = x + self.relative_position(x)
		x = self.input_conditioning(x)
		x = x.permute([1, 2, 0])
		x = self.ract1(x)
		x = x.to(self.device) # SPD brings it back to CPU
		x, shape = self.att(x)
		x = self.ract2(x)
		return x, shape
	
	def _initializePositionEncoder(self, config):
		conv = nn.Conv1d(config.in_features, config.in_features, config.position_encoder, padding=config.position_encoder // 2, groups=16)
		nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
		nn.init.constant_(conv.bias, 0)
		conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
		return nn.Sequential(conv, nn.GELU())
	
	def freeze_features(self, unfreeze=False, finetuning=False):
		for param in self.parameters():
			param.requires_grad = unfreeze
		if finetuning:
			self.mask_replacement.requires_grad = False

	
if __name__ == "__main__":


	# TODO move this to mAtt.py file
	# Cannot currently due to import issues
	
	input = torch.randn(3, 19, 1000)
	output = E2R(epochs=3)(input)
	print(output.shape)
