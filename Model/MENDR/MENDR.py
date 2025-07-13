import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.mAtt.spd import SPDTangentSpace

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class MENDR_model(nn.Module):
    def __init__(self, autoencoder, combined_contextualizer, device, wavelet_contextualizer=None, contextualizer_size="LARGE"):
        assert isinstance(autoencoder, MENDRPatchEncoder), f"Autoencoder must be of type MENDRPatchEncoder, but got {type(autoencoder)}"
        assert contextualizer_size in ["LARGE", "TINY"], f"Contextualizer size must be either 'LARGE' or 'TINY', but got {contextualizer_size}"
        if wavelet_contextualizer is not None:
            assert contextualizer_size == "LARGE", "Wavelet contextualizer is only supported for LARGE size"

        super().__init__()
        self.device = device
        self.contextualizer_size=contextualizer_size
        self.encoder = autoencoder
        self.combined_contextualizer = combined_contextualizer
        if wavelet_contextualizer is not None:
            self.wavelet_contextualizer = wavelet_contextualizer
            self.mendr_contextualizer = MENDRContextualizerLarge(self.wavelet_contextualizer, self.combined_contextualizer).to(device)


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

        #print(patchified_inputs['delta'])
        '''
        for band, inp in patchified_inputs.items():
            assert not torch.isnan(inp).any(), f"Band: {band}"
        for band, inp in encodings.items():
            # this was returning true before??
            assert not torch.isnan(inp).any(), f"Band: {band} {inp} {patchified_inputs[band]}"
        for band, inp in decodings.items():
            assert not torch.isnan(inp).any(), f"Band: {band}"
        '''
        batch_size = patchified_inputs['delta'].shape[0]
        patch_num = patchified_inputs['delta'].shape[1]
        if self.contextualizer_size == "TINY":
            _, output, _ = self.combined_contextualizer(encodings, batch_size, patch_num, mask_ratio=0.0)
            #assert not torch.isnan(output).any()
            return patchified_inputs, encodings, decodings, None, output
        elif self.contextualizer_size == "LARGE":
            combined_manifold_output, wavelet_manifold_output, _ = self.mendr_contextualizer(encodings, batch_size, patch_num) # Mask indices should never be used here
            return patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output

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
