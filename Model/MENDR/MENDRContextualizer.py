import torch
import torch.nn as nn


'''
BENDR-style Contextualizer using mATT module
'''
class mATTContextualizer(nn.Module):
	def __init__(self, config):
		super(mATTContextualizer, self).__init__()
		self.config = config
		self.mATT = mATT(config)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, input, mask):
		return self.mATT(input, mask)