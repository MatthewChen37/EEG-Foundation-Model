import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MENDREncoder import MENDRWindowEncoder
from MENDRContextualizer import MENDRContextualizer

class MENDR(nn.Module):

    def __init__(self):
        self.mendr_encoder = MENDRWindowEncoder(device=device)
        self.mendr_contextualizer = MENDRContextualizer(device=device, epochs=4, num_channels=19)
        
        # Initialize temperature as a trainable parameter
        self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True), requires_grad=True)
        self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
        self.contrastive_loss_fn_combined = nn.MSELoss()

    def forward(self, data):
        relevant_bands = [data[band] for band in BANDS]
        inputs = dict(zip(BANDS, relevant_bands))
        
        encoder_output = self.encoder(data['graph'], inputs)
        wavelet_manifold_output, epoched_shape = self.contextualizer.WaveletContextualizer(encoder_output)

        return encoder_output, wavelet_manifold_output