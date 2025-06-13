import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random, os, tqdm, mlflow
from Model.baseModelTrainer import BaseModelTrainer
from Datasets.datasetTUAB import WaveletTUABDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	roc_auc_score,
	precision_recall_curve,
	auc,
	f1_score,
	cohen_kappa_score
)
	
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class MENDRFinetuner(BaseModelTrainer):
    def __init__(self, MENDR, Decoder, optimizer, cfg, **kwargs):
        if cfg.training_params.task_loss == 'BCEWithLogitsLoss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task loss function: {cfg.training_params.task_loss}")

        super(MENDRFinetuner, self).__init__(optimizer=optimizer,
            cfg=cfg,
            mendr_model=MENDR,  
            decoder=Decoder,
            task_loss_fn=self.loss_fn,
            lr=cfg.training_params.learning_rate,
			l2_weight_decay=cfg.training_params.l2_weight_decay,
            metrics=dict(),
            ckpt_dir=cfg.training_params.ckpt_dir, **kwargs)
    
    def forward(self, data):
        '''
		Looks similar to MENDR_model forward
		but is modified for downstream finetuning
		'''
        graphs = data['graph']
        patchified_inputs = self.mendr_model._super_patchify(data)
        encodings, decodings = self.mendr_model.mendr_encoder(graphs, patchified_inputs)
        batch_size = patchified_inputs['delta'].shape[0]
        patch_num = patchified_inputs['delta'].shape[1]
        patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output = model(data['graph'], inputs)
        return patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output

    def backward(self, task_loss):
        self.optimizer.zero_grad()
        task_loss.backward()
    
    def fit(self, training_dataset, cfg, validation_dataset=None):
        self.epoch = 0
        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset
        training_dataloader, validation_dataloader = self._setup_experiment(cfg)

        for epoch in range(cfg.training_params.epochs):
            epoch_metrics = {}
            self.epoch = epoch
            ### TRAINING ###
            train_pbar = tqdm.trange(len(training_dataloader), desc="Epoch {}".format(epoch), ncols=300, position=0, leave=True)
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

            if validation_dataloader != None:
                self.train(False)
                pbar = tqdm.trange(len(validation_dataloader), desc="Predicting", ncols=300, position=0, leave=True)
                val_data_iterator = iter(validation_dataloader)
                for iteration in pbar:
                    input_batch = self._get_batch(val_data_iterator)
                    val_metrics = self.evaluate_step(input_batch)
                    pbar.set_postfix(val_metrics)
                    epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")

            ### SAVE ###
            if not self.scheduler_after_batch:
                self.optimizer.scheduler_step_cosine_annealing()
            
        mlflow.end_run()
        if cfg.meta_params.log_model_params_and_grads:
            self.logger.closeWriter()

        return self.mendr_model, self.decoder

    def train_step(self, inputs):
        self.train(True)
        patchified_inputs, encodings, _, _, combined_manifold_output = self.mendr_model(inputs)
        prediction = self.decoder(combined_manifold_output).float()
        # Check for NaNs
        assert not torch.isnan(prediction).any()
        ground_truth = inputs['graph'].y.to(self.device).float()
        task_loss = self.loss_fn(prediction.squeeze(), ground_truth)
        with torch.no_grad():
            metrics = self.calculate_metrics(prediction, ground_truth)

        self.backward(task_loss)
        self.optimizer.step()
        metrics['lr'] = self.optimizer.scheduler.get_last_lr()[0]
        return metrics

    def evaluate_step(self, inputs):
        self.train(False)
        patchified_inputs, encodings, _, _, combined_manifold_output = self.mendr_model(inputs)
        prediction = self.decoder(combined_manifold_output).float()
        ground_truth = inputs['graph'].y.to(self.device).float()
        with torch.no_grad():
            metrics = self.calculate_metrics(prediction, ground_truth)
        return metrics

    def calculate_metrics(self, prediction, ground_truth):
        if self.cfg.dataset_params.task == 'binary':
            metrics = self._calculate_metrics_binary(prediction, ground_truth)
        return metrics

    def _calculate_metrics_binary(self, prediction, ground_truth):
        metrics = {}
        score_y = torch.sigmoid(prediction)
        pred_y = torch.gt(score_y, 0.5).long().cpu().numpy()
        score_y = score_y.cpu().numpy()

        for metric in self.cfg.dataset_params.metrics:
            if metric == 'accuracy':
                metrics['accuracy'] = accuracy_score(ground_truth.cpu().numpy(), pred_y)
            elif metric == 'balanced_accuracy':
                metrics['balanced_accuracy'] = balanced_accuracy_score(ground_truth.cpu().numpy(), pred_y)
            elif metric == 'auc_pr':
                precision, recall, thresholds = precision_recall_curve(ground_truth.cpu().numpy(), score_y, pos_label=1)
                metrics['auc_pr'] = auc(recall, precision)
            elif metric == 'auroc':
                metrics['auroc'] = roc_auc_score(ground_truth.cpu().numpy(), score_y)
        return metrics

    def test_one_epoch(model, mode='Validating'):
        task_loss_sum = 0
        recon_loss_sum = 0
        self.mendr_model.eval()
        self.decoder.eval()

        pbar = tqdm.trange(len(self.finetune_test_loader), desc=mode)
        data_iterator = iter(self.finetune_test_loader)
        balanced_accuracies = []

        truths = []
        preds = []
        with torch.no_grad():
            for iteration in pbar:
                data = next(data_iterator)
                patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output = self.forward(data)
                encodings_relevant_patch = dict()
                for band, encoding in encodings.items():
                    encodings_relevant_patch[band] = encoding[:, 1, :, :][:, None, :, :]
                #prediction = decoder(encodings_relevant_patch).float()
                '''
                wavelet_manifold_output_relevant_patch = dict()
                for band, wavelet_manifold in wavelet_manifold_output.items():
                    wavelet_manifold_output_relevant_patch[band] = wavelet_manifold[:, 1, :, :][:, None, :, :]
                '''

                prediction = decoder(encodings_relevant_patch, combined_manifold_output[:, 1, :, :][:, None, :, :]).float()
                '''
                prediction = decoder(combined_manifold_output[:, 1, :, :][:, None, :, :]).float()
                prediction = decoder(combined_manifold_output).float()
                #prediction = decoder(encodings_relevant_patch, combined_manifold_output[:, 1, :, :][:, None, :, :]).float()
                #prediction = decoder(encodings, combined_manifold_output).float()

                prediction = decoder(encodings_relevant_patch,
                                    wavelet_manifold_output_relevant_patch,
                                    combined_manifold_output[:, 1, :, :][:, None, :, :]).float()

                '''
                ground_truth = data['graph'].y.to(device).long()

                #ground_truth_one_hot = nn.functional.one_hot(ground_truth, num_classes=6).float()

                #task_loss = task_loss_fn(prediction, ground_truth_one_hot)
                task_loss = task_loss_fn(prediction, ground_truth)

                delta_loss, theta_loss, alpha_loss, beta_loss, gamma_loss = self._get_recon_loss(patchified_inputs, decodings, recon_loss_fn)
                pred_y = torch.max(prediction, dim=-1)[1]
                truths += ground_truth.clone().cpu().squeeze().numpy().tolist()
                preds += pred_y.clone().cpu().squeeze().numpy().tolist()

                task_loss_sum += task_loss.item()
                recon_loss = delta_loss.item() + theta_loss.item() + alpha_loss.item() + beta_loss.item() + gamma_loss.item()
                recon_loss_sum += recon_loss

                curr_balanced_accuracy = balanced_accuracy_score(ground_truth.clone().cpu().squeeze().numpy(), pred_y.clone().cpu().squeeze().numpy())
                curr_f1 = f1_score(ground_truth.clone().cpu().squeeze().numpy(), pred_y.clone().cpu().squeeze().numpy(), average='weighted')
                curr_cohen_kappa = cohen_kappa_score(ground_truth.clone().cpu().squeeze().numpy(), pred_y.clone().cpu().squeeze().numpy())
                balanced_accuracies.append(curr_balanced_accuracy)

                pbar.set_postfix(task_loss=task_loss.item(), recon_loss=recon_loss, balanced_accuracy=f'{curr_balanced_accuracy*100:.3f}', F1=f'{curr_f1*100:.3f}', Cohens_Kappa=f'{curr_cohen_kappa*100:.3f}', lr=optimizer.scheduler.get_last_lr()[0])
            truths = np.array(truths)
            preds = np.array(preds)
            average_balanced_accuracy = np.mean(balanced_accuracies)
            balanced_accuracy = balanced_accuracy_score(truths, preds)
            f1 = f1_score(truths, preds, average='weighted')
            cohens_kappa = cohen_kappa_score(truths, preds)
            print(f'Mean Acc: {balanced_accuracy} F1 Score: {f1} Average Average Acc: {average_balanced_accuracy} Cohen Kappa: {cohens_kappa}')
        return task_loss_sum, recon_loss_sum, balanced_accuracy, f1, cohens_kappa
