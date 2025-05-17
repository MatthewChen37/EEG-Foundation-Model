import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MENDRAutoencoderTrainer(BaseModelTrainer):
    '''
	Based on BENDRTrainer.py	
	'''
    def __init__(self, MENDRAutoencoder, config, **kwargs):
        self.reconstruction_loss_function = nn.MSELoss()
        super(MENDRAutoencoderTrainer, self).__init__(autoencoder=MENDRAutoencoder,
                                                    reconstruction_loss_fn=self.reconstruction_loss_function,
                                                    lr=config.learning_rate,
                                                    l2_weight_decay=config.l2_weight_decay,
                                                    metrics=dict(),
                                                    ckpt_dir=config.ckpt_dir,
                                                    **kwargs)
    def forward(self, data):
        graphs = data['graph']
        patchified_inputs = self.autoencoder._super_patchify(data)
