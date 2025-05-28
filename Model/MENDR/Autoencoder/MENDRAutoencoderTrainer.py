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
        self.num_recons = cfg.meta_params.num_recons
        super(MENDRAutoencoderTrainer, self).__init__(autoencoder=MENDRAutoencoder, 
                                                    optimizer=optimizer,
                                                    cfg=cfg,
                                                    **kwargs)

    def forward(self, data):
        patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
        return patchified_inputs, encodings, decodings

    def backward(self, loss_dict):
        self.optimizer.zero_grad()
        combined_loss = sum(loss_dict.values())
        combined_loss.backward()
        '''
        for band in loss_dict:
            loss_dict[band].backward()
        '''

    def train_step(self, inputs):
        self.train(True)
        patchified_inputs, encodings, decodings = self.forward(inputs)
        loss_dict = WaveletReconstructionLoss(patchified_inputs, decodings)
        self.backward(loss_dict)
        self.optimizer.step()
        metrics = {band: loss.item() for band, loss in loss_dict.items()}
        metrics['lr'] = self.optimizer.scheduler.get_last_lr()[0]
        return metrics
    
    def evaluate_step(self, inputs, step_idx):
        self.train(False)
        patchified_inputs, encodings, decodings = self.forward(inputs)

        loss_dict = WaveletReconstructionLoss(patchified_inputs, decodings)
        if step_idx == 0:
            recon_enc = dict()
            recon_dec = dict()
            for band in BANDS: # Just look at patches from first sample/subject
                decodings[band] = decodings[band].view(patchified_inputs[band].shape)
                recon_enc[band] = patchified_inputs[band][0, :self.num_recons].detach().cpu().numpy()
                recon_dec[band] = decodings[band][0, :self.num_recons].detach().cpu().numpy()
            fig = plotReconstruction(recon_enc, recon_dec, f"epoch_{self.epoch} wavelet patch reconstructions")
            mlflow.log_figure(fig, f"epoch_{self.epoch}_reconstruction.pdf")
            plt.close(fig)

        return {band: loss.item() for band, loss in loss_dict.items()}

    def fit(self, training_dataset, cfg, validation_dataset=None):
        self.epoch = 0
        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset
        training_dataloader, validation_dataloader = self._setup_experiment(cfg)

        for epoch in range(cfg.training_params.epochs):
            epoch_metrics = {}
            self.epoch = epoch
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
                if cfg.meta_params.log_model_params_and_grads:
                    self.logger.log_model_gradients(self.autoencoder, epoch=epoch*len(train_pbar) + iteration)
                if self.scheduler_after_batch:
                    self.optimizer.scheduler_step_cosine_annealing()
            
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
            if cfg.meta_params.save_model:
                self._retain_best(epoch, epoch_metrics)
            self.standard_logging(epoch_metrics, "End of Epoch")
            if cfg.meta_params.log_model_params_and_grads: 
                self.logger.logEncoderParams(self.autoencoder, step=epoch)
            mlflow.log_metrics(epoch_metrics, step=epoch)
            if not self.scheduler_after_batch:
                self.optimizer.scheduler_step_cosine_annealing()
        if cfg.meta_params.save_final_model:
            self._retain_best(epoch, epoch_metrics)
        mlflow.end_run()

        if cfg.meta_params.log_model_params_and_grads:
            self.logger.closeWriter()
    
    def _retain_best(self, epoch_idx : int, metrics_to_check: dict):
        tqdm.tqdm.write("Retaining checkpoint...")
        epoch_ckpt_dir = f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}_{epoch_idx}'
        self.save_best(epoch_ckpt_dir)
        print(f"Saved Model to: {self.ckpt_dir}/{mlflow.active_run().info.run_id}_{self.epoch}")
        # Always save scheduler 
        torch.save(self.optimizer.scheduler.state_dict(), f'{epoch_ckpt_dir}/scheduler.pth')
        self.load_best(epoch_ckpt_dir)