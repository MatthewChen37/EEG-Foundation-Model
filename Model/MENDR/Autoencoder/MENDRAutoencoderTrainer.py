import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...baseModelTrainer import BaseModelTrainer
from ..WaveletLoss import WaveletReconstructionLoss
from Explainability.plotReconstruction import plotReconstruction
from Model.loggingUtil import MENDRLogger
import matplotlib.pyplot as plt
import mlflow
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRAutoencoderTrainer(BaseModelTrainer):
    '''
	Based on BENDRTrainer.py	
	'''
    def __init__(self, MENDRAutoencoder, optimizer, cfg, **kwargs):
        self.reconstruction_loss_function = nn.MSELoss()
        self.recon_loss_pref = {
            'delta': 1.0,
            'theta': 1.0,
            'alpha': 1.0,
            'beta':  1.0,
            'gamma': 1.0,
        }
        super(MENDRAutoencoderTrainer, self).__init__(autoencoder=MENDRAutoencoder, 
                                                    optimizer=optimizer,
                                                    cfg=cfg,
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

    def fit(self, training_dataset, cfg, validation_dataset=None):
        self.epoch = 0
        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset
        training_dataloader, validation_dataloader = self._setup_experiment(cfg)

        '''
        ### TRAINING ###
        train_pbar = tqdm.trange(len(training_dataloader), desc="Epoch {}".format(epoch), ncols=400, position=0, leave=True)
        train_data_iterator = iter(training_dataloader)
        self.train(True)
        for iteration in train_pbar:
            input_batch = self._get_batch(train_data_iterator)
            train_metrics = self.train_step(input_batch)
            train_pbar.set_postfix(train_metrics)
            mlflow.log_metrics(train_metrics, step=epoch*len(train_pbar) + iteration)
            epoch_metrics = self._epoch_metrics(epoch_metrics, train_metrics, "training")
            if self.scheduler_after_batch:
                self.optimizer.scheduler_step(epoch*len(train_pbar) + iteration)
            # Logging 
            self._retain_best(epoch, epoch_metrics)
            self.standard_logging(epoch_metrics, "End of Epoch")
            self.logger.log_model_gradients(self.mendr_model.mendr_encoder, epoch=epoch * len(train_pbar) + iteration)

        ### VALIDATION ###
        if validation_dataloader != None:
            self.train(False)
            pbar = tqdm.trange(len(validation_dataloader), desc="Predicting", ncols=400, position=0, leave=True)
            val_data_iterator = iter(validation_dataloader)
            for iteration in pbar:
                input_batch = self._get_batch(val_data_iterator)
                val_metrics = self.evaluate_step(input_batch, iteration)
                epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")
                pbar.set_postfix(val_metrics)

        ### SAVE ###
        self._retain_best(epoch, epoch_metrics)
        self.standard_logging(epoch_metrics, "End of Epoch")
        self.logger.logEncoderParams(self.mendr_model.mendr_encoder, step=epoch)
        mlflow.log_metrics(epoch_metrics, step=epoch)
        print("Epoch: ", epoch, "Total Training Loss: ", epoch_metrics['total_epoch_training_Combined Riemannian Loss'],
                                "Total Validation Loss: ", epoch_metrics['total_epoch_validation_Combined Riemannian Loss'])
        if self.ckpt_dir != None:
                print(f"Saved Model to: {self.ckpt_dir}/{mlflow.active_run().info.run_id}_{self.epoch}_{self.mendr_model.contextualizer_size.upper()}")
        if not self.scheduler_after_batch:
            self.optimizer.scheduler_step(epoch)
        '''
        mlflow.end_run()
        self.logger.closeWriter()

    
    def _epoch_metrics(self, aggregated_metrics, metric_dict, step):
        for metric in metric_dict:
            if metric != 'lr':
                if metric not in aggregated_metrics :
                    aggregated_metrics[f'total_epoch_{step}_{metric}'] = metric_dict[metric]
                else:
                    aggregated_metrics[f'total_epoch_{step}_{metric}'] += metric_dict[metric]
        return aggregated_metrics