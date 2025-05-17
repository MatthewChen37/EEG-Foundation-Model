import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...baseModelTrainer import BaseModelTrainer
from ..WaveletLoss import WaveletReconstructionLoss
from ....Explainability.plotReconstruction import plotReconstruction
import matplotlib.pyplot as plt
import mlflow

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRAutoencoderTrainer(BaseModelTrainer):
    '''
	Based on BENDRTrainer.py	
	'''
    def __init__(self, MENDRAutoencoder, config, **kwargs):
        self.reconstruction_loss_function = nn.MSELoss()
        self.recon_loss_pref = {
            'delta': 1.0,
            'theta': 1.0,
            'alpha': 1.0,
            'beta':  1.0,
            'gamma': 1.0,
        }

        super(MENDRAutoencoderTrainer, self).__init__(autoencoder=MENDRAutoencoder,
                                                    reconstruction_loss_fn=self.reconstruction_loss_function,
                                                    lr=config.learning_rate,
                                                    l2_weight_decay=config.l2_weight_decay,
                                                    metrics=dict(),
                                                    ckpt_dir=config.ckpt_dir,
                                                    **kwargs)
    def forward(self, data):
        patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
        return patchified_inputs, encodings, decodings

    def backward(self, loss_dict):
        self.optimizer.zero_grad()
        for band in loss_dict:
            loss_dict[band].backward(retain_graph=False)

    def train_step(self, inputs):
        self.train(True)
        patchified_inputs, encodings, decodings = self.forward(inputs)
        loss_dict = WaveletReconstructionLoss(patchified_inputs, decodings, self.recon_loss_pref, loss_strategy='sum')
        self.backward(loss_dict)
        return {band: loss.item() for band, loss in loss_dict.item}
    
    def evaluate_step(self, inputs, step_idx):
        NUM_RECONS = 4
        self.train(False)
        patchified_inputs, encodings, decodings = self.forward(inputs)
        loss_dict = WaveletReconstructionLoss(patchified_inputs, decodings, self.recon_loss_pref, loss_strategy='sum')
        if step_idx == 0:
            recon_enc = dict()
            recon_dec = dict()
            for band in BANDS: # Just look at patches from first sample/subject
                decodings[band] = decodings[band].view(patchified_inputs[band].shape)
                recon_enc[band] = patchified_inputs[band][0, :NUM_RECONS].detach().cpu().numpy()
                recon_dec[band] = decodings[band][0, :NUM_RECONS].detach().cpu().numpy()

            fig = plotReconstruction(patchified_inputs, recon_dec, f"epoch_{self.epoch} reconstructions")
            mlflow.log_figure(fig, f"epoch_{self.epoch}_reconstruction.pdf")
            plt.close(fig)

        return {band: loss.item() for band, loss in loss_dict.items()}

    def fit