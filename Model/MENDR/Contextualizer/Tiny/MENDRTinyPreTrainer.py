import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Explainability.embeddingVisualization import plotSPDEmbedding
import mlflow
import matplotlib.pyplot as plt
import tqdm

ABS_PRECISION = 10
REL_PRECISION = 10
class MENDRTinyPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, contextualizer, optimizer, cfg, **kwargs):
		self.mask_ratio = cfg.training_params.mask_ratio
		self.svd = SVD.apply
		self.contrastive_loss_fn = nn.MSELoss()

		# Freeze the autoencoder and disable the decoder
		for param in autoencoder.parameters():
			param.requires_grad = False
		autoencoder.eval()
		for band, encoder_decoder in autoencoder.encoder_decoders.items():
			encoder_decoder.disableDecoder()

		assert isinstance(contextualizer, MENDRContextualizerTiny), f"Contextualizer must be of type MENDRContextualizerTiny, but got {type(contextualizer)}"

		super(MENDRTinyPreTrainer, self).__init__(autoencoder=autoencoder, contextualizer=contextualizer,
			optimizer=optimizer,
			cfg=cfg,
			**kwargs)

		# Clamp gradients
		# This clips gradients before backpropagation: https://stackoverflow.com/a/54816498
		for p in self.parameters():
			if p.requires_grad:
				p.register_hook(lambda grad: torch.clamp(grad,
				-cfg.training_params.gradient_clip_value,
				cfg.training_params.gradient_clip_value))


	def forward(self, data):
		'''
		Looks similar to MENDR_model forward
		but is modified for the contrastive learning task(s)
		'''
		_, encodings, _ = self.autoencoder.forward(data)

		# Combined contrastive loss
		riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = self._epochMaskedRecon(encodings, self.contrastive_loss_fn)

		return {
				'encodings': encodings,
				'combined_manifold_output': combined_manifold_output,
				'combined_manifold_output_masked': combined_manifold_output_masked,
				'riemannian_loss': riemannian_loss,
				'mask_idxes': mask_idxes
		}

	def backward(self, loss):
		self.optimizer.zero_grad()
		loss.backward()

	def train_step(self, inputs):
		self.train(True)
		outputs = self.forward(inputs)
		self.backward(loss=outputs['riemannian_loss'])
		self.optimizer.step()
		train_metrics = self._calculate_metrics(outputs['riemannian_loss'].item())
		return train_metrics

	def evaluate_step(self, inputs, step_idx):
		self.train(False)
		with torch.no_grad():
			outputs = self.forward(inputs)
			eval_metrics = self._calculate_metrics(outputs['riemannian_loss'].item())

			if step_idx == 0: # Log only the first batch in the validation set
				combined_manifold_output = outputs['combined_manifold_output']
				combined_manifold_output_masked = outputs['combined_manifold_output_masked']
				batch_size, num_patches, _, _ = combined_manifold_output.shape
				combined_manifold_output = combined_manifold_output.reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
				combined_manifold_output_masked = combined_manifold_output_masked.reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
				_, combined_fig = plotSPDEmbedding(None, combined_manifold_output, combined_manifold_output_masked, inputs['subject_name'], outputs['mask_idxes'], num_patches=num_patches, num_cols=num_patches)
				mlflow.log_figure(combined_fig, f"epoch_{self.epoch}_combined_embeddings.pdf")
				#plt.close(combined_fig)
		return eval_metrics

	def fit(self, training_dataset, cfg, validation_dataset=None):
		self.epoch = 0
		self.train_dataset = training_dataset
		self.validation_dataset = validation_dataset
		training_dataloader, validation_dataloader = self._setup_experiment(cfg, rank=0)

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
					self.logger.log_model_gradients(self.contextualizer, epoch=epoch*len(train_pbar) + iteration)
				if self.scheduler_after_batch:
					self.optimizer.scheduler_step_cosine_annealing()

			### VALIDATION ###
			if validation_dataset != None:
				self.train(False)
				pbar = tqdm.trange(len(validation_dataloader), desc="Validation", ncols=400, position=0, leave=True)
				val_data_iterator = iter(validation_dataloader)
				for iteration in pbar:
					input_batch = self._get_batch(val_data_iterator)
					val_metrics = self.evaluate_step(input_batch, step_idx=iteration)
					pbar.set_postfix(val_metrics)
					if cfg.meta_params.log_model_params_and_grads:
						self.logger.log_model_gradients(self.contextualizer, epoch=epoch*len(pbar) + iteration)
				epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")
				mlflow.log_metrics(epoch_metrics, step=epoch)

			### SAVE ###
			if cfg.meta_params.save_model:
				self._retain_best(epoch, epoch_metrics)
				self.standard_logging(epoch_metrics, "End of Epoch")
			if cfg.meta_params.log_model_params_and_grads: 
				self.logger.logContextualizerParams(self.contextualizer, step=epoch)
				self.logger.logMENDRTrainerParams(None, self.contextualizer.Contextualizer.mask, step=epoch)
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


	def _calculate_metrics(self, riemannian_loss):
		return {
			'riemannian_loss': riemannian_loss,
			'lr': f"{self.optimizer.scheduler.get_last_lr()[0]:.7f}",
		}
		
	def _epochMaskedRecon(self, wavelet_manifold_output, criterion):
		batch_size = wavelet_manifold_output['delta'].shape[0]
		num_epochs = wavelet_manifold_output['delta'].shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output, combined_manifold_output_masked, mask_idxes, _ = self.contextualizer(wavelet_manifold_output, batch_size, num_epochs, mask_ratio=self.mask_ratio)
		# of shape Batch, epoch, C, C

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		og_eigenvalues = torch.log(torch.linalg.svdvals(combined_manifold_output[mask_idxes]))
		masked_eigenvalues = torch.log(torch.linalg.svdvals(combined_manifold_output_masked[mask_idxes]))
		#print(og_eigenvalues[0:5])
		#print(masked_eigenvalues[0:5])
		riemannian_loss = 100*criterion(og_eigenvalues, masked_eigenvalues)
		'''
		for batch_idx in range(batch_size):
			for i in range(2):
				assert not torch.allclose(og_eigenvalues[batch_idx*4 + i], og_eigenvalues[batch_idx*4 + (i + 1)], atol=1e-12, rtol=1e-08), f"Eigenvalues are the same: {og_eigenvalues[batch_idx*4 + i]} {og_eigenvalues[batch_idx*4 + (i + 1)]}"
				assert not torch.allclose(masked_eigenvalues[batch_idx*4 + i], masked_eigenvalues[batch_idx*4 + (i + 1)], atol=1e-8, rtol=1e-05), f"Eigenvalues are the same: {masked_eigenvalues[batch_idx*4 + i]} {masked_eigenvalues[batch_idx*4 + (i + 1)]}"
		'''
		#print("Og Eigenvalues: ", og_eigenvalues)
		#print("Masked Eigenvalues: ", masked_eigenvalues)
		return riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes

	# Debugging function, usually not used.	
	def _plot_embeddings_train(self, combined_manifold_output, combined_manifold_output_masked, mask_idxes):
		subject_names = ['0', '1', '2', '3']
		if self.mendr_model.contextualizer_size.upper() == 'LARGE':
			combined_manifold_output= combined_manifold_output.clone().detach().reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
			combined_manifold_output_masked = combined_manifold_output_masked.clone().detach().reshape(-1, combined_manifold_output.shape[-2], combined_manifold_output.shape[-1])
			wavelet_figs, combined_fig = plotSPDEmbedding(None, combined_manifold_output, combined_manifold_output_masked, subject_names, mask_idxes)
			for band, wavelet_fig in wavelet_figs.items():
				mlflow.log_figure(wavelet_fig, f"train_epoch_{self.epoch}_{band}_wavelet_embeddings.pdf")
				plt.close(wavelet_fig)
			mlflow.log_figure(combined_fig, f"train_epoch_{self.epoch}_combined_embeddings.pdf")