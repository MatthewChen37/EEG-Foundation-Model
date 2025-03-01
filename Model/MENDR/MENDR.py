import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MENDREncoder import MENDRWindowEncoder
from MENDRContextualizer import MENDRContextualizer

class MENDR(nn.Module):

    def __init__(self, device, epochs=6, num_channels=19):
        self.mendr_encoder = MENDRWindowEncoder(device=device)
        self.mendr_contextualizer = MENDRContextualizer(device=device, epochs=epochs, num_channels=num_channels)
        
        # Initialize temperature as a trainable parameter
        self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True), requires_grad=True)
        self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
        self.contrastive_loss_fn_combined = nn.MSELoss()

    def forward(self, data):
        relevant_bands = [data[band] for band in BANDS]
        inputs = dict(zip(BANDS, relevant_bands))
        encoder_output = self.encoder(data['graph'], inputs)
        combined_manifold_output, wavelet_manifold_output = self.contextualizer(encoder_output)
        return encoder_output, wavelet_manifold_output, combined_manifold_output

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
