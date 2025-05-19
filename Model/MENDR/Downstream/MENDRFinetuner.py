import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from ..baseModelTrainer import BaseModelTrainer
from Datasets.datasetTUAB import WaveletTUABDataset

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
def MENDRFinetuner(BaseModelTrainer):
    def __init__(self, MENDR, Decoder, config, Adaptor=None, **kwargs):
        seed = config.seed
        ### Seed ###
        torch.cuda.empty_cache()
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print("*" * 50)
        recon_loss_fn = nn.MSELoss()
        if config.dataset == "TUAB":
            loss_fn = nn.BCEWithLogitsLoss()
            finetune_train_dataset = WaveletTUABDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUAB-128Hz/train", frac=1.0)
            finetune_eval_dataset = WaveletTUABDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUAB-128Hz/eval", frac=1.0)
        print("Dataset Loaded. Length of Train Dataset: ", len(finetune_train_dataset))
        print("Dataset Loaded. Length of Validation Dataset: ", len(finetune_eval_dataset))

        self.finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=128, num_workers=8, shuffle=True, persistent_workers=True)
        self.finetune_eval_loader = DataLoader(finetune_eval_dataset, batch_size=128, num_workers=8, shuffle=True, persistent_workers=True)
        print(f"Length of Train Dataset: {len(self.finetune_train_loader)}")
        print(f"Length of Validation Dataset: {len(self.finetune_eval_loader)}")

        super(MENDRFinetuner, self).__init__(mendr_model=MENDR, 
            decoder=Decoder,
            adaptor=Adaptor,
            recon_loss_fn=recon_loss_fn,
            task_loss_fn=loss_fn,
            lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay,
            metrics=dict(),
            ckpt_dir=config.ckpt_dir, **kwargs)

        
    def _fft_loss(self, output, target, loss_fn):
        hanning_window = torch.hann_window(output.shape[-1]).to(device)
        output_windowed = output * hanning_window.expand_as(output)
        target_windowed = target * hanning_window.expand_as(target)

        output_fft = torch.fft.fft(output_windowed, dim=-1)
        output_amplitude = torch.abs(output_fft)
        output_angle = torch.angle(output_fft)

        target_fft = torch.fft.fft(target_windowed, dim=-1)
        target_amplitude = torch.abs(target_fft)
        target_angle = torch.abs(target_fft)

        return (loss_fn(output_amplitude, target_amplitude) + (loss_fn(output_angle, target_angle)))

    def calculate_band_loss(self, inputs, outputs, loss_fn, loss_type='real'):
        outputs_reshaped = outputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
        if loss_type == 'real':
            loss = loss_fn(inputs, outputs_reshaped)
        elif loss_type == 'fourier':
            loss = _fft_loss(output_reshaped, inputs, loss_fn)
        else:
            loss = loss_fn(inputs, outputs_reshaped) + _fft_loss(output_reshaped, inputs, loss_fn) * 1e-2
        return loss

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

    def _get_recon_loss(self, patchified_inputs, decodings, recon_loss_fn):
        delta_loss = self._calculate_band_loss(patchified_inputs['delta'], decodings['delta'], recon_loss_fn)
        theta_loss = self._calculate_band_loss(patchified_inputs['theta'], decodings['theta'], recon_loss_fn)
        alpha_loss = self._calculate_band_loss(patchified_inputs['alpha'], decodings['alpha'], recon_loss_fn)
        beta_loss = self._calculate_band_loss(patchified_inputs['beta'],   decodings['beta'], recon_loss_fn)
        gamma_loss = self._calculate_band_loss(patchified_inputs['gamma'], decodings['gamma'], recon_loss_fn)
        return delta_loss, theta_loss, alpha_loss, beta_loss, gamma_loss

    def _get_mtl_backward_params(self, encodings):
        shared_features = [encoding for band, encoding in encodings.items()]
        task_params = [self.decoder.parameters()]
        for band in BANDS:
            model_params = self.mendr_model.mendr_encoder.encoder_decoders[band].getDecoderParams()
            task_params.append(model_params)
        return shared_features, task_params


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
