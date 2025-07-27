import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge, MENDRWaveletContextualizer, MENDRCombinedContextualizer
import from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian
import mlflow
import matplotlib.pyplot as plt
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRLargeCombinedPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, wavelet_contextualizer, combined_contextualizer, optimizer, cfg, **kwargs):
		self.svd = SVD.apply
		self.contrastive_loss_fn = nn.CrossEntropyLoss()

		# Freeze the autoencoder and disable the decoder
		for param in autoencoder.parameters():
			param.requires_grad = False
		autoencoder.eval()
		for band, encoder_decoder in autoencoder.encoder_decoders.items():
			encoder_decoder.disableDecoder()

		# Freeze the wavelet contextualizer
		for param in wavelet_contextualizer.parameters():
			param.requires_grad = False
		wavelet_contextualizer.eval()

		assert isinstance(wavelet_contextualizer, MENDRWaveletContextualizer), f"Wavelet Contextualizer must be of type MENDRWaveletContextualizer, but got {type(wavelet_contextualizer)}"
		assert isinstance(combined_contextualizer, MENDRCombinedContextualizer), f"Combined Contextualizer must be of type MENDRombinedContextualizer, but got {type(combined_contextualizer)}"

		super(MENDRLargeCombinedPreTrainer, self).__init__(autoencoder=autoencoder, wavelet_contextualizer=wavelet_contextualizer,
			combined_contextualizer=combined_contextualizer,
			contrastive_loss_fn_combined=self.contrastive_loss_fn,
			lr=cfg.train_params.learning_rate,
			l2_weight_decay=cfg.training_params.l2_weight_decay,
			metrics=dict(),
			ckpt_dir=cfg.training_params.ckpt_dir,
			**kwargs)

		# Clamp gradients
		# This clips gradients before backpropagation: https://stackoverflow.com/a/54816498
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -cfg.training_params.gradient_clip_value, cfg.training_params.gradient_clip_value))

	def backward(self, loss):
		self.optimizer.zero_grad()
		loss.backward()

	def forward(self, data):
		_, encodings, _ = self.autoencoder.forward(data)
		batch_size = encodings['delta'].shape[0]
		num_patches = encodings['delta'].shape[1]
		wavelet_manifold_output = self.wavelet_contextualizer(encodings, batch_size=batch_size, num_patches=num_patches)
		combined_manifold_output, combined_manifold_output_masked, mask_idxes = self.combined_contextualizer(wavelet_manifold_output, batch_size=batch_size, num_patches=num_patches)
		riemannian_loss = self._epochMaskedRecon(combined_manifold_output, combined_manifold_output_masked, mask_idxes, self.contrastive_loss_fn_combined)
		return {
				'encodings': encodings,
				'combined_manifold_output': combined_manifold_output,
				'combined_manifold_output_masked': combined_manifold_output_masked,
				'riemannian_loss': riemannian_loss,
		}
	
	def train_step(self, data):
		self.optimizer.zero_grad()
		outputs = self.forward(data)
		self.backward(loss=outputs['riemannian_loss'])
		self.optimizer.step()
		return self._calculate_metrics(outputs['riemannian_loss'])
	
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
		return eval_metrics


	def _calculate_metrics(self, riemannian_loss):
		metrics = {
			'riemannian_loss': riemannian_loss,
			'lr': f"{self.optimizer.scheduler.get_last_lr()[0]:.7f}",
		}
		return metrics
		
	def epochMaskedRecon(self, combined_manifold_output, combined_manifold_output_masked, mask_idxes, criterion):
		og_eigenvalues = torch.log(torch.linalg.svdvals(combined_manifold_output[mask_idxes]))
		masked_eigenvalues = torch.log(torch.linalg.svdvals(combined_manifold_output_masked[mask_idxes]))
		riemannian_loss = 100*criterion(og_eigenvalues, masked_eigenvalues)
		return riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes
	

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
