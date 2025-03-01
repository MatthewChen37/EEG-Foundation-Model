import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MENDREncoder import MENDRAutoEncoder
from MENDRContextualizer import MENDRContextualizer

class MENDR(nn.Module):

    def __init__(self):
        self.mendr_encoder = MENDRAutoEncoder(device=device)
        self.mendr_contextualizer = MENDRContextualizer(device=device, epochs=4, num_channels=19)

        # Initialize temperature as a trainable parameter
		self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True), requires_grad=True)
		self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
		self.contrastive_loss_fn_combined = nn.MSELoss()
		self.negatives_loo = 20

		# Mask is a learnable SPD matrix
		# We indirectly optimize on the SPD manifold because by Cholesky Decomposition 
		# X * X.T is always SPD
		self.mask = torch.from_numpy(np.random.rand(contextualizer.channels, contextualizer.channels))
		self.mask = nn.Parameter(self.mask, requires_grad=True)
