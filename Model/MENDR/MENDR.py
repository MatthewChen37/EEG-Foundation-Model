import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.MENDR.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.mAtt.spd import SPDTangentSpace

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class MENDR_model(nn.Module):
    def __init__(self, device, temp, num_channels=19, 
                sampling_rate=128, hop_length=0.5,
                delta_encoded_h=19,
                theta_encoded_h=19,
                alpha_encoded_h=38,
                beta_encoded_h=76,
                gamma_encoded_h=114,
                high_encoded_h=152,
                super_patch_seconds=10,
                contextualizer_size="LARGE"):
        '''
        Sampling rate is in Hertz
        Hop Length analogous to BIOT's hop length parameter but is specified in seconds.
        By default 0.5 seconds.  
        '''
        super().__init__()
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.super_patch_seconds = super_patch_seconds
        self.device = device
        self.contextualizer_size=contextualizer_size
        self.num_channels = num_channels

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
        
        if self.contextualizer_size.upper() == "LARGE":
            self.mendr_contextualizer = MENDRContextualizerLarge(
                delta_encoded_h=self.encoded_h['delta'],
                theta_encoded_h=self.encoded_h['theta'],
                alpha_encoded_h=self.encoded_h['alpha'],
                beta_encoded_h=self.encoded_h['beta'],
                gamma_encoded_h=self.encoded_h['gamma'],
                high_encoded_h=self.encoded_h['high'],
                temp=temp,
                device=device)
        elif self.contextualizer_size.upper() == "TINY":
            encoded_h_total = self.encoded_h['delta'] + self.encoded_h['theta'] + self.encoded_h['alpha'] + self.encoded_h['beta'] + self.encoded_h['gamma']
            self.mendr_contextualizer = MENDRContextualizerTiny(encoded_h=encoded_h_total, device=device)
        else:
            raise ValueError("Contextualizer size must be either 'LARGE' or 'TINY'")

        self.tangent_space = SPDTangentSpace(self.mendr_contextualizer.encoded_out, device=device)
        
        
    def forward(self, graphs, data):
        patchified_inputs = self._super_patchify(data)
        batch_size = patchified_inputs['delta'].shape[0]
        patch_num = patchified_inputs['delta'].shape[1]
        encodings, decodings = self.mendr_encoder(graphs, patchified_inputs) # this is the shared module in torch JD
        combined_manifold_output, wavelet_manifold_output, _ = self.mendr_contextualizer(encodings, batch_size, patch_num) # Mask indices should never be used here
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