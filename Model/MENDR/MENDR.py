import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.MENDR.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.MENDRContextualizerTiny import MENDRContextualizerTiny

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class MENDR_model(nn.Module):
    def __init__(self, device, num_channels=19, 
                sampling_rate=128, hop_length=0.5,
                delta_encoded_h=38,
                theta_encoded_h=38,
                alpha_encoded_h=38,
                beta_encoded_h=38,
                gamma_encoded_h=76,
                high_encoded_h=76,
                super_patch_seconds=10,
                temp=10.0,
                contextualizer_size="LARGE"):
        '''
        Sampling rate is in Hertz
        Hop Length analogous to BIOT's hop length parameter but is specified in seconds.
        By default 0.5 seconds.  
        '''
        super(MENDR_model, self).__init__()
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.super_patch_seconds = super_patch_seconds
        self.device = device

        # Each represents one second of data
        self.SUPPORTED_WAVELET_LENGTHS = {
            128 : {
                'delta': 4, # 0-4Hz, 1/32 
                'theta': 4, # 4-8Hz, 1/32
                'alpha': 8, # 8-16Hz, 1/16
                'beta': 16, # 16-32Hz, 1/8
                'gamma': 32, # 32-64Hz, 1/4
                'high': 64, # 64-128Hz, 1/2
            }, 
            256 : { 
                #TODO
            }
        }

        self.WAVELET_LENGTHS = self.SUPPORTED_WAVELET_LENGTHS[self.sampling_rate]

        self.WAVELET_SUPER_PATCH_LENGTHS = dict()
        self.WAVELET_SUPER_PATCH_HOP_LENGTHS = dict()
        for band, second_length in self.WAVELET_LENGTHS.items():
            self.WAVELET_SUPER_PATCH_LENGTHS[band] = self.super_patch_seconds * second_length
            self.WAVELET_SUPER_PATCH_HOP_LENGTHS[band] = int(self.WAVELET_SUPER_PATCH_LENGTHS[band] * self.hop_length)


        self.encoded_h = {
            'delta': delta_encoded_h,
            'theta': theta_encoded_h,
            'alpha': alpha_encoded_h,
            'beta': beta_encoded_h,
            'gamma': gamma_encoded_h,
            'high': high_encoded_h
        }
            
        self.mendr_encoder = MENDRPatchEncoder(
            num_channels=19,
            delta_sub_patch_size=self.WAVELET_LENGTHS['delta'],
            theta_sub_patch_size=self.WAVELET_LENGTHS['theta'],
            alpha_sub_patch_size=self.WAVELET_LENGTHS['alpha'],
            beta_sub_patch_size=self.WAVELET_LENGTHS['beta'],
            gamma_sub_patch_size=self.WAVELET_LENGTHS['gamma'],
            high_sub_patch_size=self.WAVELET_LENGTHS['high'],
            delta_encoded_h=self.encoded_h['delta'],
            theta_encoded_h=self.encoded_h['theta'],
            alpha_encoded_h=self.encoded_h['alpha'],
            beta_encoded_h=self.encoded_h['beta'],
            gamma_encoded_h=self.encoded_h['gamma'],
            high_encoded_h=self.encoded_h['high'],
            delta_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['delta'],
            theta_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['theta'],
            alpha_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['alpha'],
            beta_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['beta'],
            gamma_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['gamma'],
            high_super_patch_seq_len=self.WAVELET_SUPER_PATCH_LENGTHS['high'],
            device=device)
        
        if contextualizer_size.upper() == "LARGE":
            self.mendr_contextualizer = MENDRContextualizerLarge(device=device)
        elif contextualizer_size.upper() == "TINY":
            # TODO: Modify
            self.mendr_contextualizer = MENDRContextualizerTiny(device=device)
        else:
            raise ValueError("Contextualizer size must be either 'LARGE' or 'TINY'")
        
        # Initialize temperature as a trainable parameter
        self.temp1 = torch.nn.Parameter(torch.tensor(temp, requires_grad=True), requires_grad=True)


    def forward(self, graphs, data):
        batch_size = data['delta'].shape[0]
        patchified_inputs = self._super_patchify(data)
        encodings, decodings = self.mendr_encoder(graphs, patchified_inputs)

        # Reshape encodings before passing into contextualizer
        for band in BANDS:
            if band in encodings:
                encodings[band] = encodings[band].reshape(batch_size, -1, self.encoded_h[band], self.WAVELET_SUPER_PATCH_LENGTHS[band])
        combined_manifold_output, wavelet_manifold_output, _ = self.mendr_contextualizer(encodings) # Mask indices should never be used here
        return patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output

    def _super_patchify(self, data):
        patchified_data = dict()
        for band, second_length in self.WAVELET_LENGTHS.items():
            if band in data: # Sometimes we exclude HIGH
                wavelet_data = data[band] # [Batches, Channels, Time Steps]
                patched_wavelet_data = []
                wavelet_super_patch_length = self.WAVELET_SUPER_PATCH_LENGTHS[band]
                wavelet_super_patch_hop_length = self.WAVELET_SUPER_PATCH_HOP_LENGTHS[band]
                for i in range(0, wavelet_data.shape[-1], wavelet_super_patch_hop_length):
                    if i + wavelet_super_patch_length <= wavelet_data.shape[-1]:
                        patched_wavelet_data.append(wavelet_data[:, :, i:i + wavelet_super_patch_length])
                patched_wavelet_data = torch.stack(patched_wavelet_data, dim=1) # [Batch, Patches, C, T]
                patchified_data[band] = patched_wavelet_data
        return patchified_data
                
    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze