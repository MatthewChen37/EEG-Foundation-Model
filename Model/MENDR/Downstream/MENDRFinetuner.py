import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
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
    def __init__(self, MENDR, Decoder, cfg, **kwargs):
        if cfg.training_params.task_loss == 'BCEWithLogitsLoss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task loss function: {cfg.training_params.task_loss}")

        super(MENDRFinetuner, self).__init__(mendr_model=MENDR, 
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
    

    def fit(self, training_dataset, cfg, validation_dataset=None):
        return self.mendr_model, self.decoder


    def train_one_epoch(self):
        task_loss_sum = 0
        recon_loss_sum = 0
        self.mendr_model.train()
        self.decoder.train()

        pbar = tqdm.trange(len(self.finetune_train_loader), desc="Training")
        data_iterator = iter(self.finetune_train_loader)
        balanced_accuracies = []

        for iteration in pbar:
            data = next(data_iterator)
            patchified_inputs, encodings, decodings, wavelet_manifold_output, combined_manifold_output = self.forward(data)
            
            prediction = self.decoder(combined_manifold_output).float()
            ground_truth = data['graph'].y.to(device).float()
            ground_truth = torch.repeat_interleave(ground_truth, patchified_inputs['delta'].shape[1])

            task_loss = self.task_loss_fn(prediction, ground_truth[:, None])
            delta_loss, theta_loss, alpha_loss, beta_loss, gamma_loss = self._get_recon_loss(patchified_inputs, decodings, recon_loss_fn)

            with torch.no_grad():
                score_y = torch.sigmoid(prediction)
                pred_y = torch.gt(score_y, 0.5).long().cpu().numpy()
                score_y = score_y.cpu().numpy()

                balanced_accuracy = balanced_accuracy_score(ground_truth.cpu().numpy(), pred_y)
                accuracy = accuracy_score(ground_truth.cpu().numpy(), pred_y)
                roc_auc = roc_auc_score(ground_truth.cpu().numpy(), score_y)
                precision, recall, thresholds = precision_recall_curve(ground_truth.cpu().numpy(), score_y, pos_label=1)
                pr_auc = auc(recall, precision)
            balanced_accuracies.append(balanced_accuracy)

            shared_features, task_params = self._get_mtl_backward_params(encodings)
            optimizer.zero_grad()
            #task_loss.backward()
            mtl_backward(losses=[task_loss, delta_loss, theta_loss, alpha_loss, beta_loss, gamma_loss], features=shared_features, aggregator=aggregator, tasks_params=task_params)
            optimizer.step()
            optimizer.scheduler_step(epoch * len(train_loader) + iteration)
            task_loss_sum += task_loss.item()
            recon_loss = delta_loss.item() + theta_loss.item() + alpha_loss.item() + beta_loss.item() + gamma_loss.item()
            recon_loss_sum += recon_loss
            pbar.set_postfix(task_loss=task_loss.item(), recon_loss=recon_loss, accuracy=f'{accuracy*100:.3f}', balanced_accuracy=f'{balanced_accuracy*100:.3f}', AUROC=f'{roc_auc*100:.3f}', AUCPR=f'{pr_auc*100:.3f}', lr=optimizer.scheduler.get_last_lr()[0])
        balanced_accuracies = np.array(balanced_accuracies)	
        print(f'Mean Acc: {balanced_accuracies.mean()} Acc Std: {balanced_accuracies.std()}')
        return task_loss_sum, recon_loss_sum

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
