import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random, os, tqdm, mlflow
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
        ''' 
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad,
                                                -cfg.training_params.gradient_clip_value,
                                                cfg.training_params.gradient_clip_value))
        '''
    def forward(self, inputs):
        '''
		Looks similar to MENDR_model forward
		but is modified for downstream finetuning
		'''
        patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output = self.mendr_model(inputs)
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
        if len(prediction.shape) == 3: # ISRUC
            prediction = prediction.reshape(-1, prediction.shape[-1])
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
            if len(prediction.shape) == 3: # ISRUC
                prediction = prediction.reshape(-1, prediction.shape[-1])
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

    def evaluate(self, cfg, validation_dataset):
        self.train(False)
        finetune_eval_loader = self._make_dataloader(self.validation_dataset,
                                                    cfg,
                                                    training=False,
                                                    sampler=None)
        print("*" * 100)
        print("Perform Final Evaluation")
        print("*" * 100)

        self.mendr_model.eval()
        self.decoder.eval()

        encoding_outputs = dict()
        decoding_outputs = dict()
        combined_outputs = []
        output_labels = []
        predictions = []
        final_embeddings = []

        pbar = tqdm.trange(len(finetune_eval_loader), desc="Validating")
        data_iterator = iter(finetune_eval_loader)

        with torch.no_grad():
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output = self.forward(inputs)
                '''
                for band, encoding in encodings.items():
                    print(band, encoding.shape)
                '''
                prediction = self.decoder(combined_manifold_output).float()
                if self.cfg.training_params.task_loss == 'BCEWithLogitsLoss':
                    ground_truth = inputs['graph'].y.to(self.device).float()
                elif self.cfg.training_params.task_loss == 'CrossEntropyLoss':
                    ground_truth = inputs['graph'].y.to(self.device).long()
                else:
                    raise ValueError(f"Unsupported task loss function: {self.cfg.training_params.task_loss}")

                if len(prediction.shape) == 3: # ISRUC
                    prediction = prediction.reshape(-1, prediction.shape[-1])

                loss = self.loss_fn(prediction.squeeze(), ground_truth)
                pbar.set_postfix(loss=loss.item())
                output_labels.append(ground_truth)

                if cfg.dataset_params.name == "ISRUC":
                    first_dim = combined_manifold_output.shape[0]
                    batch_size = first_dim // self.cfg.training_params.seq_of_patch_len
                    num_patches = combined_manifold_output.shape[1]
                    embedding_dim = combined_manifold_output.shape[2]

                    combined_vector = self.decoder.tangent(combined_manifold_output.view(batch_size*self.cfg.training_params.seq_of_patch_len*num_patches, embedding_dim, embedding_dim))
                    combined_vector = combined_vector.reshape(batch_size, self.cfg.training_params.seq_of_patch_len, num_patches, combined_vector.shape[-1])
                    x = self.decoder.flatten(combined_vector.clone())
                    x = x.reshape(batch_size, self.cfg.training_params.seq_of_patch_len, 15*3*7)
                    final_embedding = self.decoder.seq(x)
                    prediction = self.decoder.final_decoder(final_embedding.clone()).float()
                else:
                    batch_size = combined_manifold_output.shape[0]
                    num_patches = combined_manifold_output.shape[1]
                    embedding_dim = combined_manifold_output.shape[2]
                    x = combined_manifold_output.reshape(batch_size*num_patches, embedding_dim, embedding_dim)
                    combined_vector = self.mendr_model.combined_contextualizer.tangent_space(x)
                    combined_vector = combined_vector.reshape(batch_size, num_patches, -1)
                    x = self.decoder.flatten(combined_vector.clone())
                    final_embedding = self.decoder.seq(x)
                    prediction = self.decoder.final_decoder(final_embedding.clone()).float()

                for band, wavelet_manifold in wavelet_manifold_output.items():
                    if band not in encoding_outputs:
                        encoding_outputs[band] = []
                    if len(wavelet_manifold.shape) == 3:
                        wavelet_manifold = wavelet_manifold.reshape(batch_size, num_patches, embedding_dim, embedding_dim)
                        wavelet_manifold = torch.flatten(wavelet_manifold, start_dim=1)

                    encoding_outputs[band].append(wavelet_manifold.cpu().numpy())
                    
                for band, decoding in decodings.items():
                    if band not in decoding_outputs:
                        decoding_outputs[band] = []
                    decoding_outputs[band].append(decodings)

                combined_outputs.append(combined_vector)
                final_embeddings.append(final_embedding)

                predictions.append(prediction.cpu().numpy())

        combined_outputs = torch.cat(combined_outputs, dim=0)
        final_embeddings = torch.cat(final_embeddings, dim=0)

        encoding_output_eval_all = dict()
        for band, encodings in encoding_outputs.items():
            encoding_output_eval_all[band] = np.concatenate(encodings, axis=0)

        combined_output_eval_all = combined_outputs.cpu().numpy()
        final_embeddings_eval_all = final_embeddings.cpu().numpy()
        output_labels_eval_all = torch.cat(output_labels, dim=0).cpu().numpy().astype(np.int8)
        predictions = np.concatenate(predictions, axis=0)

        self._plot_umap(cfg, encoding_output_eval_all, output_labels_eval_all,
                predictions, final_embeddings_eval_all)

        '''
        # Plot confusion matrix
        print("*" * 100)
        print("Plotting Confusion Matrix")
        print("*" * 100)
        self._plot_confusion_matrix(cfg, output_labels_eval_all, predictions)
        '''

        mlflow.end_run()

        
    def _plot_umap(self, cfg, encoding_output_eval_all, output_labels_eval_all,
                predictions, final_embeddings_eval_all):
        reducer = umap.UMAP(n_components=2)
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        if cfg.patch_encoder_params.include_high:
            bands.append('high')
        bands.append('combined')

        for band in tqdm.tqdm(bands):
            if band == 'combined':
                if cfg.dataset_params.name == "ISRUC":
                    final_embeddings_eval_all = final_embeddings_eval_all.reshape(-1, final_embeddings_eval_all.shape[-1])
                embeddings = reducer.fit_transform(final_embeddings_eval_all, job=-1)
            else:
                data = encoding_output_eval_all[band].reshape(encoding_output_eval_all[band].shape[0], -1)
                embeddings = reducer.fit_transform(data, jobs=-1)
            fig = plt.figure(figsize=(24, 12))
            ax = fig.add_subplot()
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=[sns.color_palette()[x] for x in output_labels_eval_all])
            fig.gca().set_aspect('equal', 'datalim')
            fig.suptitle(f'{cfg.dataset_params.name} UMAP projection {band}', fontsize=24)
            self._add_legend(cfg, ax)
            mlflow.log_figure(fig, f'{cfg.dataset_params.name}_{cfg.meta_params.contextualizer_size}_umap_{band}.pdf')

        embeddings = reducer.fit_transform(final_embeddings_eval_all, job=-1)
        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot()
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=[sns.color_palette()[x] for x in output_labels_eval_all])
        fig.gca().set_aspect('equal', 'datalim')
        fig.suptitle(f'{cfg.dataset_params.name} UMAP projection Final Embeddings', fontsize=24)
        self._add_legend(cfg, ax)
        mlflow.log_figure(fig, f'{cfg.dataset_params.name}_{cfg.meta_params.contextualizer_size}_umap_final_embeddings.pdf')

    def _add_legend(self, cfg, ax):
        if cfg.dataset_params.name == 'TUAB':
            patches = [
                Patch(facecolor=sns.color_palette()[0], label='Normal'),
                Patch(facecolor=sns.color_palette()[1], label='Abnormal'),
            ]
            ax.legend(handles=patches, loc='upper right')
        elif cfg.dataset_params.name == 'TUEV':
            patches = [
                Patch(facecolor=sns.color_palette()[0], label='spsw'),
                Patch(facecolor=sns.color_palette()[1], label='gped'),
                Patch(facecolor=sns.color_palette()[2], label='pled'),
                Patch(facecolor=sns.color_palette()[3], label='eyem'),
                Patch(facecolor=sns.color_palette()[4], label='artf'),
                Patch(facecolor=sns.color_palette()[5], label='bckg'),
            ]
            ax.legend(handles=patches, loc='upper right')
        elif cfg.dataset_params.name == 'ISRUC':
            patches = [
                Patch(facecolor=sns.color_palette()[0], label='Stage 1'),
                Patch(facecolor=sns.color_palette()[1], label='Stage 2'),
                Patch(facecolor=sns.color_palette()[2], label='Stage 3'),
                Patch(facecolor=sns.color_palette()[3], label='Stage 4'),
                Patch(facecolor=sns.color_palette()[4], label='Stage 5')
            ]
            ax.legend(handles=patches, loc='upper right')

    '''    
    def _plot_confusion_matrix(self, cfg, output_labels_eval_all, predictions):
        if cfg.dataset_params.name == 'TUAB':
            labels = ['Normal', 'Abnormal']
        elif cfg.dataset_params.name == 'TUEV':
            labels = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']

        cm = confusion_matrix(output_labels_eval_all.astype(int), predictions.astype(int), normalize='all')
        
        fig, ax = plt.subplots(figsize=(24, 24))

        ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{cfg.dataset_params.name} Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(include_values=True, cmap='Blues', ax=ax)
        fig = disp.figure_
        mlflow.log_figure(fig, f'{cfg.dataset_params.name}_{cfg.meta_params.contextualizer_size}_confusion_matrix.pdf')

    '''