import torch
import torch.nn as nn
import numpy as np
from .mAtt.mAtt import E2R, AttentionManifold, SPDRectified
from .mAtt.spd import SPDTangentSpace, SPDTransform
from ..layers import Permute, Flatten
import math

'''
BENDR-style Contextualizer using mATT module 
augmented from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class MENDRContextualizerTiny(nn.Module):
	def __init__(self, device, epochs=4, num_channels=19):
		super(MENDRContextualizerLarge, self).__init__()
		self.device = device
		self.epochs = epochs
		self.channels = num_channels

		self.WaveletContextualizer = MENDRWaveletContextualizer(device=self.device, epochs=self.epochs, num_channels=self.channels)
		self.CombinedContextualizer = MENDRCombinedContextualizer(device=self.device, num_channels=self.channels)



	def forward(self, x):
		return combined_manifold_output, wavelet_manifold_output
