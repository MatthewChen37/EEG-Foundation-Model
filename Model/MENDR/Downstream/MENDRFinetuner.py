import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
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
        elif cfg.training_params.task_loss == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.training_params.task_loss_params.label_smoothing)
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
    
    def fit(self, training_dataset, cfg, validation_dataset=None, train_sampler=None, val_sampler=None):
        distributed = isinstance(train_sampler, DistributedSampler)
        rank  = dist.get_rank() if distributed else 0
        world = dist.get_world_size() if distributed else 1
        self.epoch = 0
        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset
        training_dataloader, validation_dataloader = self._setup_experiment(cfg, rank, train_sampler, val_sampler)

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
                truths = []
                preds = []
                pbar = tqdm.trange(len(validation_dataloader), desc="Predicting", ncols=300, position=0, leave=True)
                val_data_iterator = iter(validation_dataloader)
                for iteration in pbar:
                    input_batch = self._get_batch(val_data_iterator)
                    prediction, ground_truth, loss = self.evaluate_step(input_batch)
                    preds += prediction.clone().detach().cpu().squeeze().numpy().tolist()
                    truths += ground_truth.clone().detach().cpu().squeeze().numpy().tolist()
                    with torch.no_grad():
                        batch_metrics = self.calculate_metrics(prediction, ground_truth)
                    pbar.set_postfix(loss=loss.item(), **batch_metrics)
                truths = torch.from_numpy(np.array(truths))
                preds = torch.from_numpy(np.array(preds))
                with torch.no_grad():
                    val_metrics = self.calculate_metrics(preds, truths)
                epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")
                mlflow.log_metrics(epoch_metrics, step=epoch)
                for metric in val_metrics:
                    print(f'Epoch: {epoch} Mean {metric}: {val_metrics[metric]}')

                if cfg.dataset_params.name == "TUAB":
                    print(f'Epoch: {epoch} Mean Acc: {val_metrics["accuracy"]} Mean auc_pr: {val_metrics["auc_pr"]} Mean auroc: {val_metrics["auroc"]}')
                elif cfg.dataset_params.name == "TUEV":
                    print(f'Epoch: {epoch} Mean Acc: {val_metrics["accuracy"]} Mean f1: {val_metrics["f1"]} Mean cohens_kappa: {val_metrics["cohens_kappa"]}')

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
        #print("SHAPE:", patchified_inputs['delta'].shape, combined_manifold_output.shape)
        #assert not torch.isnan(combined_manifold_output).any(), f"Combined Manifold contains NaN values: {combined_manifold_output}"
        prediction = self.decoder(combined_manifold_output).float()
        # Check for NaNs
        #assert not torch.isnan(prediction).any(), f"Prediction contains NaN values: {prediction}"
        if self.cfg.training_params.task_loss == 'BCEWithLogitsLoss':
            ground_truth = inputs['graph'].y.to(self.device).float()
        elif self.cfg.training_params.task_loss == 'CrossEntropyLoss':
            ground_truth = inputs['graph'].y.to(self.device).long()
        else:
            raise ValueError(f"Unsupported task loss function: {self.cfg.training_params.task_loss}")
        '''
        if self.cfg.dataset_params.name == "TUAB":
            ground_truth = torch.repeat_interleave(ground_truth, patchified_inputs['delta'].shape[1])
        '''
        task_loss = self.loss_fn(prediction.squeeze(), ground_truth)
        with torch.no_grad():
            metrics = self.calculate_metrics(prediction, ground_truth)
        self.backward(task_loss)
        self.optimizer.step()
        metrics['loss'] = task_loss.item()
        metrics['lr'] = self.optimizer.scheduler.get_last_lr()[0]
        return metrics

    def evaluate_step(self, inputs):
        self.train(False)
        with torch.no_grad():
            patchified_inputs, encodings, _, _, combined_manifold_output = self.mendr_model(inputs)
            prediction = self.decoder(combined_manifold_output).float()
            if self.cfg.training_params.task_loss == 'BCEWithLogitsLoss':
                ground_truth = inputs['graph'].y.to(self.device).float()
            elif self.cfg.training_params.task_loss == 'CrossEntropyLoss':
                ground_truth = inputs['graph'].y.to(self.device).long()
            else:
                raise ValueError(f"Unsupported task loss function: {self.cfg.training_params.task_loss}")
            task_loss = self.loss_fn(prediction.squeeze(), ground_truth)
        return prediction, ground_truth, task_loss

    def calculate_metrics(self, prediction, ground_truth):
        if self.cfg.dataset_params.task == 'binary':
            metrics = self._calculate_metrics_binary(prediction, ground_truth)
        elif self.cfg.dataset_params.task == 'multiclass':
            metrics = self._calculate_metrics_multiclass(prediction, ground_truth)
        else:
            raise Exception("Task not found")
        return metrics

    def _calculate_metrics_binary(self, prediction, ground_truth):
        metrics = {}
        score_y = torch.sigmoid(prediction)
        pred_y = torch.gt(score_y, 0.5).long().cpu().detach().numpy()
        score_y = score_y.cpu().detach().numpy()

        for metric in self.cfg.dataset_params.metrics:
            if metric == 'accuracy':
                metrics['accuracy'] = accuracy_score(ground_truth.cpu().numpy(), pred_y)
            elif metric == 'balanced_accuracy':
                metrics['balanced_accuracy'] = balanced_accuracy_score(ground_truth.cpu().numpy(), pred_y)
            elif metric == 'auc_pr':
                if len(np.unique(ground_truth.cpu().numpy())) > 1:
                    precision, recall, thresholds = precision_recall_curve(ground_truth.cpu().numpy(), score_y, pos_label=1)
                    metrics['auc_pr'] = auc(recall, precision)
                else:
                    metrics['auc_pr'] = 0.0
            elif metric == 'auroc':
                if len(np.unique(ground_truth.cpu().numpy())) > 1:
                    metrics['auroc'] = roc_auc_score(ground_truth.cpu().numpy(), score_y)
                else:
                    metrics['auroc'] = 0.0
        return metrics

    def _calculate_metrics_multiclass(self, prediction, ground_truth):
        metrics = {}
        pred_y = torch.max(prediction, dim=-1)[1].cpu().detach().numpy()

        for metric in self.cfg.dataset_params.metrics:
            if metric == 'accuracy':
                metrics['accuracy'] = accuracy_score(ground_truth.cpu().detach().numpy(), pred_y)
            elif metric == 'balanced_accuracy':
                metrics['balanced_accuracy'] = balanced_accuracy_score(ground_truth.cpu().detach().numpy(), pred_y)
            elif metric == 'f1':
                metrics['f1'] = f1_score(ground_truth.cpu().detach().numpy(), pred_y, average='weighted')
            elif metric == 'cohens_kappa':
                metrics['cohens_kappa'] = cohen_kappa_score(ground_truth.cpu().detach().numpy(), pred_y)
        return metrics

