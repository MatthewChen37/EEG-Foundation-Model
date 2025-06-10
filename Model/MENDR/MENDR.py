import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
#from Model.MENDR.Contextualizer.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.mAtt.spd import SPDTangentSpace

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class MENDR_model(nn.Module):
    def __init__(self, autoencoder, combined_contextualizer, device, wavelet_contextualizer=None, contextualizer_size="LARGE"):
        assert isinstance(autoencoder, MENDRPatchEncoder), f"Autoencoder must be of type MENDRPatchEncoder, but got {type(autoencoder)}"
        assert contextualizer_size in ["LARGE", "TINY"], f"Contextualizer size must be either 'LARGE' or 'TINY', but got {contextualizer_size}"
        if wavelet_contextualizer is not None:
            assert contextualizer_size == "TINY", "Wavelet contextualizer is only supported for TINY size"

        super().__init__()
        self.device = device
        self.contextualizer_size=contextualizer_size
        self.encoder = autoencoder
        self.combined_contextualizer = combined_contextualizer
        if wavelet_contextualizer is not None:
            self.wavelet_contextualizer = wavelet_contextualizer

        self.trainable_state = {
            'encoder': True,
            'combined_contextualizer': True,
            'wavelet_contextualizer': True,
        }

        self.tangent_space = SPDTangentSpace(self.encoder.num_channels)
        
    def forward(self, data):
        """
        Returns the patchified inputs, encodings, decodings, wavelet manifold output (if size is LARGE), and combined manifold output.
        """

        patchified_inputs, encodings, decodings = self.encoder(data)
        if self.contextualizer_size == "TINY":
            batch_size = patchified_inputs['delta'].shape[0]
            patch_num = patchified_inputs['delta'].shape[1]
            _, output, _ = self.combined_contextualizer(encodings, batch_size, patch_num, mask_ratio=0.0)
            return patchified_inputs, encodings, decodings, None, output
        elif self.contextualizer_size == "LARGE":
            combined_manifold_output, wavelet_manifold_output, _ = self.mendr_contextualizer(encodings, batch_size, patch_num) # Mask indices should never be used here
            return None

    def freeze_autoencoder(self, require_grad=False):
        for param in self.encoder.parameters():
            param.requires_grad = require_grad
        self.trainable_state['encoder'] = require_grad

    def parameters(self):
        params = []
        if self.trainable_state['encoder']:
            params += list(self.encoder.parameters())
        if self.trainable_state['combined_contextualizer']:
            params += list(self.combined_contextualizer.parameters())
        return params
