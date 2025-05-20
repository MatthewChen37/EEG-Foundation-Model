import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotReconstruction import plotReconstruction
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian, plotWaveletEmbeddingsEuclidean
import mlflow
import matplotlib.pyplot as plt

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ABS_PRECISION = 3 # Number of decimal places to consider equal
REL_PRECISION = 1 # Relative tolerance precision
class MENDRTinyPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, contextualizer, cfg, **kwargs):
		self.mask_ratio = config.mask_ratio
		self.svd = SVD.apply
		self.contrastive_loss_fn = nn.MSELoss()
		self.contrastive_loss_pref = config.contrastive_loss_pref
		self.contrastive_combined_loss_coeff = config.contrastive_combined_loss_coeff
		self.contrastive_wavelet_loss_coeff = config.contrastive_wavelet_loss_coeff

		super(MENDRTinyPreTrainer, self).__init__(autoencoder=autoencoder, contextualizer=contextualizer,
			contrastive_loss_fn=self.contrastive_loss_fn,
			lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay,
			metrics=dict(),
			ckpt_dir=config.ckpt_dir,
			**kwargs)

		if self.multi_objective_loss_balancing_strategy == "jacobian":
			self.aggregator = UPGrad(pref_vector=torch.tensor([
				self.contrastive_loss_pref,
				self.recon_loss_pref['delta'],
				self.recon_loss_pref['theta'],
				self.recon_loss_pref['alpha'],
				self.recon_loss_pref['beta'],
				self.recon_loss_pref['gamma']
			]))

		# Clamp gradients
		# This clips gradients before backpropagation: https://stackoverflow.com/a/54816498
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -config.gradient_clip_value, config.gradient_clip_value))

		''' Unused
		self.RandomGaussianNoise = RandomGaussianNoise()
		self.RandomFTSurrogate = RandomFTSurrogate(phase_noise_magnitude=0.2, random_state=config.random_state)
		'''	
	def forward(self, data):
		'''
		Looks similar to MENDR_model forward
		but is modified for the contrastive learning task(s)
		'''
		graphs = data['graph']
		patchified_inputs = self.mendr_model._super_patchify(data)
		encodings, decodings = self.mendr_model.mendr_encoder(graphs, patchified_inputs)
		batch_size = patchified_inputs['delta'].shape[0]
		patch_num = patchified_inputs['delta'].shape[1]
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)

			# Wavelet wise contrastive loss, i.e. Multi-Resolution loss
			w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)

			# Combined contrastive loss
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = self.epochMaskedRecon(wavelet_manifold_output, epoched_shape, self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'combined_manifold_output_masked': combined_manifold_output_masked,
					'riemannian_loss': riemannian_loss,
					'wavelet_manifold_output': wavelet_manifold_output,
					'wavelet_loss': w_loss,
					'wavelet_acc': w_correct / w_pairs,
					'mask_idxes': mask_idxes
			}
		elif self.mendr_model.contextualizer_size.upper() == "TINY":
			# Combined contrastive loss
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = self.epochMaskedReconTiny(encodings, self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'combined_manifold_output_masked': combined_manifold_output_masked,
					'riemannian_loss': riemannian_loss,
					'mask_idxes': mask_idxes
			}
		else:
			raise ValueError("Unidentified Contextualizer Type")

	def backward(self, shared_features, contrastive_losses, reconstruction_losses):
		self.optimizer.zero_grad()
		if self.multi_objective_loss_balancing_strategy == "jacobian":
			self._backward_jacobian(shared_features, contrastive_losses, reconstruction_losses)
		elif self.multi_objective_loss_balancing_strategy == "sum":
			self._backward_sum(contrastive_losses, reconstruction_losses)
		elif self.multi_objective_loss_balancing_strategy == "real_time":
			self._backward_real_time(contrastive_losses, reconstruction_losses)
		else:
			raise ValueError("Unidentified Multi Objective Loss Balancing Strategy")
		
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			# Only large model has temp parameter, which is used in wavelet loss
			# Clamp temperature to non-negative values
			with torch.no_grad():
				self.mendr_model.mendr_contextualizer.temp1.copy_(torch.clamp(self.mendr_model.mendr_contextualizer.temp1, min=0.0))

	def _backward_sum(self, contrastive_losses, reconstruction_losses):
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			losses = contrastive_losses[0] + contrastive_losses[1]
		else:
			losses = contrastive_losses[0] 
		for band in reconstruction_losses:
			losses = losses + reconstruction_losses[band]
		losses.backward()

	def _backward_real_time(self, contrastive_losses, reconstruction_losses):
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			losses = (contrastive_losses[0] / contrastive_losses[0].detach()) + (contrastive_losses[1] / contrastive_losses[1].detach())
		else:
			losses = (contrastive_losses[0] / contrastive_losses[0].detach())
		for band in reconstruction_losses:
			losses = losses + (reconstruction_losses[band] / reconstruction_losses[band].detach())
		losses.backward()

	def _backward_jacobian(self, shared_features, contrastive_losses, reconstruction_losses):
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			losses = [contrastive_losses[0] + contrastive_losses[1]] + list(reconstruction_losses.values())
		elif self.mendr_model.contextualizer_size.upper() == "TINY":
			losses = [self.contrastive_combined_loss_coeff * contrastive_losses[0]] + list(reconstruction_losses.values())
		else:
			raise ValueError("Unidentified Contextualizer Type")

		tasks_params = [
				list(self.mendr_model.mendr_contextualizer.parameters()),
		]
		for band in BANDS:
			model_params = self.mendr_model.mendr_encoder.encoder_decoders[band].getDecoderParams()
			tasks_params.append(model_params)

		mtl_backward(losses=losses, features=shared_features, aggregator=self.aggregator, retain_graph=False, tasks_params=tasks_params)

	def train_step(self, inputs):
		self.train(True)
		outputs = self.forward(inputs)
		recon_losses = WaveletReconstructionLoss(outputs['patchified_inputs'], outputs['decodings'], recon_loss_pref=self.recon_loss_pref, loss_strategy=self.multi_objective_loss_balancing_strategy) # Reconstruction Loss returns a dictionary
		shared_features = [encoding for band, encoding in outputs['encodings'].items()]
		if self.mendr_model.contextualizer_size.upper() == "LARGE":
			# In a future work, figure out a way to multi-level backpropagate (i.e. jacobians for the wavelet loss and riemannian loss on riemannian and wavelet loss)
			self.backward(shared_features=shared_features, contrastive_losses=[outputs['riemannian_loss'], outputs['wavelet_loss']], reconstruction_losses=recon_losses)
			train_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), outputs['wavelet_loss'].item(), outputs['wavelet_acc'], recon_losses)
			self.optimizer.step()
		elif self.mendr_model.contextualizer_size.upper() == "TINY":
			self.backward(shared_features=shared_features, contrastive_losses=[outputs['riemannian_loss']], reconstruction_losses=recon_losses)
			train_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), None, None, recon_losses)
			self.optimizer.step()
		else:
			raise ValueError("Unidentified Contextualizer Type")
		train_metrics["lr"] = f"{self.optimizer.scheduler.get_last_lr()[0]:.7f}"

		for metric in train_metrics: # Don't want tensors in the metrics
			if isinstance(train_metrics[metric], torch.Tensor):
				train_metrics[metric] = train_metrics[metric].item()
		return train_metrics

	def evaluate_step(self, inputs, step_idx):
		NUM_RECONS = 4
		self.train(False)
		with torch.no_grad():
			outputs = self.forward(inputs)
			recon_losses = WaveletReconstructionLoss(outputs['patchified_inputs'], outputs['decodings'], recon_loss_pref=self.recon_loss_pref, loss_strategy=self.multi_objective_loss_balancing_strategy) # Reconstruction Loss returns a dictionary
			if self.mendr_model.contextualizer_size.upper() == 'LARGE':
				eval_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), outputs['wavelet_loss'].item(), outputs['wavelet_acc'], recon_losses)
			elif self.mendr_model.contextualizer_size.upper() == 'TINY':
				eval_metrics = self._calculate_metrics(outputs['riemannian_loss'].item(), None, None, recon_losses)
			else:
				raise ValueError("Unidentified Contextualizer Type")

			if step_idx == 0: # Log only the first batch in the validation set
				combined_manifold_output = outputs['combined_manifold_output']
				combined_manifold_output_masked = outputs['combined_manifold_output_masked']
				batch_size = combined_manifold_output.shape[0]
				num_patches = combined_manifold_output.shape[1]
				assert outputs['mask_idxes'].shape[0] == batch_size, f"Masked Index Shape: {outputs['mask_idxes'].shape} Batch Size: {batch_size}"
				assert outputs['mask_idxes'].shape[1] == num_patches, f"Masked Index Shape: {outputs['mask_idxes'].shape} Batch Size: {batch_size}"
				assert len(inputs['subject_name']) == batch_size, f"Subject Name Length: {len(inputs['subject_name'])} Batch Size: {batch_size}"

				combined_manifold_output= combined_manifold_output.reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
				combined_manifold_output_masked = combined_manifold_output_masked.reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
				if self.mendr_model.contextualizer_size.upper() == 'LARGE':
					wavelet_manifold_output = outputs['wavelet_manifold_output']
					'''
					predictions = []
					for idx in range(combined_manifold_output_masked.shape[0]):
							predictions.append(combined_manifold_output_masked[idx])
					for predictions_idx in range(len(predictions)):
						if predictions_idx != len(predictions) - 1:
							assert torch.allclose(predictions[predictions_idx], predictions[predictions_idx + 1]) == False, f"Predictions: {predictions[predictions_idx]} and {predictions[predictions_idx + 1]} are equal: {predictions_idx}"
					'''
					wavelet_figs, combined_fig = plotSPDEmbedding(wavelet_manifold_output, combined_manifold_output, combined_manifold_output_masked, inputs['subject_name'], outputs['mask_idxes'])
					for band, wavelet_fig in wavelet_figs.items():
						mlflow.log_figure(wavelet_fig, f"epoch_{self.epoch}_{band}_wavelet_embeddings.pdf")
						plt.close(wavelet_fig)
					mlflow.log_figure(combined_fig, f"epoch_{self.epoch}_combined_embeddings.pdf")
					plt.close(combined_fig)
					fig = plotWaveletEmbeddingsRiemannian(wavelet_manifold_output,
														 combined_manifold_output.reshape(batch_size, num_patches, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out),
														 f"epoch_{self.epoch} wavelet embeddings", reduction="TSNE")
					mlflow.log_figure(fig, f"epoch_{self.epoch}_wavelet_embeddings.html")
				elif self.mendr_model.contextualizer_size.upper() == 'TINY':
					_, combined_fig = plotSPDEmbedding(None, combined_manifold_output, combined_manifold_output_masked, inputs['subject_name'], outputs['mask_idxes'])
					mlflow.log_figure(combined_fig, f"epoch_{self.epoch}_combined_embeddings.pdf")
					plt.close(combined_fig)
				else:
					raise ValueError("Unidentified Contextualizer Type")

				recon_enc = dict()
				recon_dec = dict()
				for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']: # Just look at patches from first sample/subject
					outputs['decodings'][band] = outputs['decodings'][band].view(outputs['patchified_inputs'][band].shape)
					recon_enc[band] = outputs['patchified_inputs'][band][0, :NUM_RECONS].detach().cpu().numpy()
					recon_dec[band] = outputs['decodings'][band][0, :NUM_RECONS].detach().cpu().numpy()

				fig = plotReconstruction(recon_enc, recon_dec, f"epoch_{self.epoch} reconstructions")
				mlflow.log_figure(fig, f"epoch_{self.epoch}_reconstruction.pdf")
				plt.close(fig)

			for metric in eval_metrics:
				if isinstance(eval_metrics[metric], torch.Tensor):
					eval_metrics[metric] = eval_metrics[metric].item()

			return eval_metrics

	
	def _calculate_metrics(self, combined_riemannian_loss, wavelet_loss, wavelet_acc, recon_loss):
		if wavelet_loss is None or wavelet_acc is None:
			return {
				'Combined Riemannian Loss': combined_riemannian_loss,
			} | recon_loss

		else:
			return {
				'Combined Riemannian Loss': combined_riemannian_loss,
				'Wavelet Loss': wavelet_loss,
				'Wavelet Acc': wavelet_acc,
			} | recon_loss

	def _epochMaskedRecon(self, wavelet_manifold_output, criterion):
		batch_size = wavelet_manifold_output['delta'].shape[0]
		num_epochs = wavelet_manifold_output['delta'].shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output_masked, combined_manifold_output, mask_idxes = self.mendr_model.mendr_contextualizer(wavelet_manifold_output, batch_size, num_epochs, mask_ratio=self.mask_ratio)
		# of shape Batch, epoch, C, C

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])
		return self.contrastive_combined_loss_coeff * riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes
	
	def _plot_embeddings_train(self, combined_manifold_output, combined_manifold_output_masked, mask_idxes):
		batch_size = combined_manifold_output.shape[0]
		num_patches = combined_manifold_output.shape[1]
		subject_names = ['0', '1', '2', '3']
		if self.mendr_model.contextualizer_size.upper() == 'LARGE':
			combined_manifold_output= combined_manifold_output.clone().detach().reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
			combined_manifold_output_masked = combined_manifold_output_masked.clone().detach().reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
			wavelet_figs, combined_fig = plotSPDEmbedding(None, combined_manifold_output, combined_manifold_output_masked, subject_names, mask_idxes)
			for band, wavelet_fig in wavelet_figs.items():
				mlflow.log_figure(wavelet_fig, f"train_epoch_{self.epoch}_{band}_wavelet_embeddings.pdf")
				plt.close(wavelet_fig)
			mlflow.log_figure(combined_fig, f"train_epoch_{self.epoch}_combined_embeddings.pdf")
