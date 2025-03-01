import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.MENDR.MENDREncoder import MENDRWindowEncoder
from Model.MENDR.MENDRContextualizer import MENDRContextualizer

class MENDR_model(nn.Module):
    def __init__(self, device, epochs=6, num_channels=19):
        super(MENDR_model, self).__init__()
        self.mendr_encoder = MENDRWindowEncoder(device=device)
        self.mendr_contextualizer = MENDRContextualizer(device=device, epochs=epochs, num_channels=num_channels)
        
        # Initialize temperature as a trainable parameter
        self.temp1 = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True), requires_grad=True)

    def forward(self, data):
        relevant_bands = [data[band] for band in BANDS]
        inputs = dict(zip(BANDS, relevant_bands))
        encoder_output = self.encoder(data['graph'], inputs)
        combined_manifold_output, wavelet_manifold_output = self.contextualizer(encoder_output)
        return encoder_output, wavelet_manifold_output, combined_manifold_output

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
