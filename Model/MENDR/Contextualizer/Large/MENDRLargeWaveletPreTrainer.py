import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRWaveletContextualizer
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian
import mlflow
import matplotlib.pyplot as plt
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRLargeWaveletPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, autoencoder, wavelet_contextualizer, optimizer, cfg, **kwargs):
		self.svd = SVD.apply
		self.contrastive_loss_fn = nn.CrossEntropyLoss()

		# Freeze the autoencoder and disable the decoder
		for param in autoencoder.parameters():
			param.requires_grad = False
		autoencoder.eval()
		for band, encoder_decoder in autoencoder.encoder_decoders.items():
			encoder_decoder.disableDecoder()

		assert isinstance(wavelet_contextualizer, MENDRWaveletContextualizer), f"Contextualizer must be of type MENDRWaveletContextualizer, but got {type(wavelet_contextualizer)}"

		super(MENDRLargeWaveletPreTrainer, self).__init__(
			autoencoder=autoencoder,
		 	wavelet_contextualizer=wavelet_contextualizer,
			optimizer=optimizer,
			cfg=cfg,
			**kwargs)

		# Clamp gradients
		# This clips gradients before backpropagation: https://stackoverflow.com/a/54816498
		for p in self.parameters():
			p.register_hook(lambda grad: torch.clamp(grad, -cfg.training_params.gradient_clip_value, cfg.training_params.gradient_clip_value))

	def forward(self, data):
		patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
		wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)
		return {
				'patchified_inputs': patchified_inputs,
				'encodings': encodings,
				'decodings': decodings,
				'wavelet_manifold_output': wavelet_manifold_output,
				'wavelet_loss': w_loss,
				'wavelet_acc': w_correct / w_pairs,
			}

	def backward(self, loss):
		self.optimizer.zero_grad()
		loss.backward()

	def train_step(self, inputs):
		self.train(True)
		outputs = self.forward(inputs)
		self.backward(loss=outputs['wavelet_loss'])
		self.optimizer.step()
		train_metrics = self._calculate_metrics(outputs['wavelet_loss'].item())
		return train_metrics

	def evaluate_step(self, inputs, step_idx):
		self.train(False)
		with torch.no_grad():
			outputs = self.forward(inputs)
			eval_metrics = self._calculate_metrics(outputs['wavelet_loss'].item())

			if step_idx == 0: # Log only the first batch in the validation set
				fig = plotWaveletEmbeddingsRiemannian(wavelet_manifold_output,
													combined_manifold_output.reshape(batch_size, num_patches, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out),
													f"epoch_{self.epoch} wavelet embeddings", reduction="TSNE")
				mlflow.log_figure(fig, f"epoch_{self.epoch}_wavelet_embeddings.html")

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
					self.logger.log_model_gradients(self.wavelet_contextualizer, epoch=epoch*len(train_pbar) + iteration)
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
					mlflow.log_metrics(val_metrics, step=epoch*len(pbar) + iteration)
					epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")
					if cfg.meta_params.log_model_params_and_grads:
						self.logger.log_model_gradients(self.wavelet_contextualizer, epoch=epoch*len(pbar) + iteration)

			### SAVE ###
			if cfg.meta_params.save_model:
				self._retain_best(epoch, epoch_metrics)
				self.standard_logging(epoch_metrics, "End of Epoch")
			if cfg.meta_params.log_model_params_and_grads: 
				self.logger.logContextualizerParams(self.wavelet_contextualizer, step=epoch)
				self.logger.logMENDRTrainerParams(None, self.wavelet_contextualizer.Contextualizer.mask, step=epoch)
			mlflow.log_metrics(epoch_metrics, step=epoch)

			if not self.scheduler_after_batch:
				self.optimizer.scheduler_step_cosine_annealing()
		if cfg.meta_params.save_final_model:
			self._retain_best(epoch, epoch_metrics)
		mlflow.end_run()
			
		if cfg.meta_params.log_model_params_and_grads:
			self.logger.closeWriter()

	def leave_one_out(self, embeddings, criterion, negatives=20):
		"""
		Compute leave-one-out loss for wavelet embeddings.

		Args:
			embeddings (dict): Dictionary of SPD wavelet embeddings.
			criterion: Loss function (e.g., CrossEntropyLoss).
			negatives: Number of Negatives will be negatives - 1 (more like a sub-batch selection)

		Returns:
			loss (torch.Tensor): Total leave-one-out loss.
			correct (int): Number of correct predictions.
			pairs (int): Number of prediction pairs.
		"""
		frequency_bands = list(embeddings.keys())
		num_targets = len(frequency_bands)
		batch_size = embeddings['delta'].shape[0]

		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			# Average embeddings of all other modalities
			other_embeddings = []
			negative_selection = torch.randperm(batch_size)
			negative_indices = negative_selection[:negatives]
			for j in list(range(i)) + list(range(i + 1, num_targets)):
				embedding_tensor = embeddings[frequency_bands[j]].clone().view(-1, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out) # [Batch * epochs, C, C]
				embedding_tensor = embedding_tensor[negative_indices]
				other_embeddings.append(embedding_tensor)

			curr_target = embeddings[frequency_bands[i]].clone().view(-1, self.mendr_model.mendr_contextualizer.encoded_out, self.mendr_model.mendr_contextualizer.encoded_out)[negative_indices]


			other_embeddings_mean = self.mendr_model.mendr_contextualizer.WaveletContextualizer._batch_LogEuclideanMean(other_embeddings, frequency_bands[i])

			# Why does this fail for higher precisions?
			# Answer: AttenionManifold's forward and SPDTransforms Forward are numerically unstable for FloatingPoint Precision calculations
			# I.E. it will create attention values that are approximately symmetric matrices up to a certain precision.
			# If we change the tensor's values to Double rather than float this assertion passes for PRECISION=16
			# However, to accelerate training at scale we need to keep everything as Floats
			# Note that the total number of digits a Float32 can store is around 6 to 9:
			# https://stackoverflow.com/questions/56514892/how-many-digits-can-float8-float16-float32-float64-and-float128-contain
			# assuming torch.float = np.float32
			# Potential future direction is "Quantizing" SPD matrices
			# assert torch.allclose(curr_target, curr_target.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(curr_target)
			# assert torch.allclose(other_embeddings_mean, other_embeddings_mean.mT, atol=(10 ** -ABS_PRECISION), rtol=(10 ** -REL_PRECISION)), self._findNonSymmetry(other_embeddings_mean)

			# Compute logits
			logits = self._batchWiseMatrixSimilarity(curr_target, other_embeddings_mean)
			labels = torch.arange(logits.shape[0], device=self.device)

			# Forward loss
			forward_logits = logits
			l = criterion(forward_logits, labels)
			loss += l
			correct += (torch.argmax(forward_logits, axis=0) == labels).sum().item()
			pairs += forward_logits.size(0)

			# Reverse loss - Ensure logits are symmetric
			reverse_logits = logits.T
			l = criterion(reverse_logits, labels)
			loss += l
			correct += (torch.argmax(reverse_logits, axis=0) == labels).sum().item()
			pairs += reverse_logits.size(0)
		return loss, correct, pairs

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B, eps=1e-12):
		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		# Based on the Log-Euclidean metric
		a_u, a_s, a_v = self.svd(batch_A + eps)
		b_u, b_s, b_v = self.svd(batch_B + eps)
		tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s + eps)) @ a_v.permute(0, 2, 1)
		tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s + eps)) @ b_v.permute(0, 2, 1)
		inner_term = tensor_log_A[:, None, :, :] - tensor_log_B[None, :, :, :]
		output = torch.linalg.matrix_norm(inner_term, ord='fro')

		output = 1 / (1 + torch.log(1 + output))
		output = output * torch.exp(self.wavelet_contextualizer.temp1)
		return output