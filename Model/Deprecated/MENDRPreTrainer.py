import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from ..baseModelTrainer import BaseModelTrainer
from ..transforms import RandomTemporalCrop, RandomGaussianNoise, RandomFTSurrogate
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch
from .mAtt import StiefelParameter
from .safeSVD import SVD, svdv2
from scipy.linalg import orth
from ..MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge
from ..MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotReconstruction import plotReconstruction
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian, plotWaveletEmbeddingsEuclidean
import mlflow
import matplotlib.pyplot as plt
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ABS_PRECISION = 3 # Number of decimal places to consider equal
REL_PRECISION = 1 # Relative tolerance precision
class MENDRPreTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, MENDR, config, **kwargs):
		self.negatives_loo = config.negatives_loo
		self.mask_ratio = config.mask_ratio
		self.svd = SVD.apply
		self.contrastive_loss_fn_wavelet = nn.CrossEntropyLoss()
		self.contrastive_loss_fn_combined = nn.MSELoss()
		self.contrastive_loss_pref = config.contrastive_loss_pref
		self.contrastive_combined_loss_coeff = config.contrastive_combined_loss_coeff
		self.contrastive_wavelet_loss_coeff = config.contrastive_wavelet_loss_coeff

		assert config.multi_objective_loss_balancing_strategy in ["sum", "real_time", "jacobian"], f"Invalid multi_objective_loss_balancing_strategy: {args.multi_objective_loss_balancing_strategy}"
		self.multi_objective_loss_balancing_strategy = config.multi_objective_loss_balancing_strategy

		self.recon_loss_pref = {
			'delta': config.delta_reconstructive_loss_pref,
			'theta': config.theta_reconstructive_loss_pref,
			'alpha': config.alpha_reconstructive_loss_pref,
			'beta': config.beta_reconstructive_loss_pref,
			'gamma': config.gamma_reconstructive_loss_pref
		}
		
		super(MENDRPreTrainer, self).__init__(mendr_model=MENDR, contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
			contrastive_loss_fn_combined = self.contrastive_loss_fn_combined, lr=config.learning_rate,
			l2_weight_decay=config.l2_weight_decay, metrics=dict(), ckpt_dir=config.ckpt_dir, **kwargs)

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

	# Useful for debugging, only called on assertion error
	def _findNonSymmetry(self, A):
		torch.set_printoptions(precision=ABS_PRECISION + REL_PRECISION, threshold=200, linewidth=1e2)
		if len(A.shape) == 2:
			non_symmetries = torch.where(A != A.T)
			output = f"Indices: {non_symmetries} OgVals: {A[non_symmetries]} TransposeVals: {A.T[non_symmetries]}, Num_indices: {non_symmetries[0].shape}"
		else: # Batch of matrices
			non_symmetries = torch.where(A != A.mT)
			output = f"Indices: {non_symmetries} OgVals: {A[non_symmetries]} TransposeVals: {A.mT[non_symmetries]}, Num_indices: {non_symmetries[0].shape}"
		return output

	def epochMaskedRecon(self, wavelet_manifold_output, epoched_shape, criterion):
		frequency_bands = list(wavelet_manifold_output.keys())
		batch_size = epoched_shape[0]
		num_epochs = epoched_shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output = self.mendr_model.mendr_contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(wavelet_manifold_output)

		combined_manifold_output_masked = combined_manifold_output.clone()
		combined_manifold_output_masked, mask_idxes = self.mendr_model.mendr_contextualizer.CombinedContextualizer(combined_manifold_output_masked, epoched_shape, mask_ratio=self.mask_ratio)
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])
		'''
		# For debugging
		predictions = []
		for batch_idx in range(combined_manifold_output_masked.shape[0]):
			for patch_idx in range(combined_manifold_output_masked.shape[1]):
					if mask_idxes[batch_idx, patch_idx] == True:
						predictions.append(combined_manifold_output_masked[batch_idx, patch_idx])
		for predictions_idx in range(len(predictions)):
			if predictions_idx != len(predictions) - 1:
				mask = torch.matmul(self.mendr_model.mendr_contextualizer.CombinedContextualizer.mask, self.mendr_model.mendr_contextualizer.CombinedContextualizer.mask.T)
				assert torch.allclose(predictions[predictions_idx], predictions[predictions_idx + 1]) == False, f"Predictions: {predictions[predictions_idx]} and {predictions[predictions_idx + 1]} are equal, {mask}"
				assert torch.allclose(predictions[predictions_idx], mask) == False, f"Predictions: {predictions[predictions_idx]} and {mask} are equal"

		print("Shapes:")
		print(torch.masked_select(combined_manifold_output_masked, mask_idxes.to(self.device)[:, :, None, None]).shape)
		print(torch.masked_select(combined_manifold_output, mask_idxes.to(self.device)[:, :, None, None]).shape)
		'''
		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])
		'''	
		if torch.is_grad_enabled():
			self._plot_embeddings_train(combined_manifold_output, combined_manifold_output_masked, mask_idxes)
		'''
		return self.contrastive_combined_loss_coeff * riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes

	def epochMaskedReconTiny(self, wavelet_manifold_output, criterion):
		batch_size = wavelet_manifold_output['delta'].shape[0]
		num_epochs = wavelet_manifold_output['delta'].shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output_masked, combined_manifold_output, mask_idxes = self.mendr_model.mendr_contextualizer(wavelet_manifold_output, batch_size, num_epochs, mask_ratio=self.mask_ratio)
		# of shape Batch, epoch, C, C

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])
		return self.contrastive_combined_loss_coeff * riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes

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
			#other_embeddings_mean = torch.stack(other_embeddings, dim=0).sum(0) / (num_targets - 1)


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
			#logits = torch.matmul(curr_target, other_embeddings_mean.T) * torch.exp(self.mendr_model.mendr_contextualizer.temp1)
			#logits = (logits + logits.T) * 0.5
			#logits = torch.matmul(curr_target, other_embeddings_mean.T) * torch.exp(self.mendr_model.mendr_contextualizer.temp1)
			labels = torch.arange(logits.shape[0], device=self.device)

			#print("Logits:", logits)
			#print("Labels:", labels)
			#print("Predictions:", torch.argmax(logits, axis=0))
			#print("Predictions2:", torch.argmax(logits, axis=1))
			#print("Temperature:", self.mendr_model.mendr_contextualizer.temp1)
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
		return self.contrastive_wavelet_loss_coeff * loss, correct, pairs

	'''
	Currently not being used
	'''
	def check_degenerate_singular_values(self, matrix, tol=1e-5):
		"""
		Checks for degenerate singular values in a matrix.

		Args:
			matrix (torch.Tensor): The input matrix.
			tol (float): Tolerance for considering singular values as degenerate.

		Returns:
			bool: True if degenerate singular values are found, False otherwise.
		"""

		singular_values = torch.linalg.svdvals(matrix)
		diff = torch.abs(singular_values[:, :-1] - singular_values[:, 1:])
		return torch.any(diff < tol), torch.where(diff < tol)

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B, eps=1e-12):
		# This can be sped up
		output = torch.zeros((batch_A.shape[0], batch_B.shape[0])).to(self.device)
		# Based on the Log-Euclidean metric
		'''
		batch_A_rank = torch.linalg.matrix_rank(batch_A)
		batch_B_rank = torch.linalg.matrix_rank(batch_B)
		if (batch_A_rank != batch_A.shape[1]).any():
			raise Exception("batch A not full rank")
		if (batch_B_rank != batch_B.shape[1]).any():
			raise Exception("batch B not full rank")

		badA, whereA = self.check_degenerate_singular_values(batch_A)
		print(whereA[0].shape, whereA[1].shape, torch.unique(whereA[0]).shape)

		if badA:
			print(whereA)
			raise Exception("batch A degenerate")

		if self.check_degenerate_singular_values(batch_B):
			raise Exception("batch B degenerate")
		
		a_u = torch.zeros(batch_A.shape).to(self.device)
		a_s = torch.zeros(batch_A.shape[0], batch_A.shape[1]).to(self.device)
		a_v = torch.zeros(batch_A.shape).to(self.device)

		b_u = torch.zeros(batch_B.shape).to(self.device)
		b_s = torch.zeros(batch_B.shape[0], batch_B.shape[1]).to(self.device)
		b_v = torch.zeros(batch_B.shape).to(self.device)

		for i in range(batch_A.shape[0]):
			a_u[i], a_s[i], a_v[i] = self.svd(batch_A[i])
			b_u[i], b_s[i], b_v[i] = self.svd(batch_B[i])

		'''
		a_u, a_s, a_v = self.svd(batch_A + eps)
		b_u, b_s, b_v = self.svd(batch_B + eps)
		tensor_log_A = a_u @ torch.diag_embed(torch.log(a_s + eps)) @ a_v.permute(0, 2, 1)
		tensor_log_B = b_u @ torch.diag_embed(torch.log(b_s + eps)) @ b_v.permute(0, 2, 1)
		inner_term = tensor_log_A[:, None, :, :] - tensor_log_B[None, :, :, :]
		output = torch.linalg.matrix_norm(inner_term, ord='fro')

		output = 1 / (1 + torch.log(1 + output))
		output = output * torch.exp(self.mendr_model.mendr_contextualizer.temp1)
		return output