import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import ptwt
from ..baseModelTrainer import BaseModelTrainer
from .MENDRContextualizer import _make_mask, _make_span_from_seeds
from .WaveletLoss import WaveletReconstructionLoss
from torch_geometric.utils import unbatch

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
PRECISION = 7 # Number of decimal places to consider equal

class MENDRTrainer(BaseModelTrainer):
	'''
	Based on BENDRTrainer.py	
	'''
	def __init__(self, encoder, contextualizer, r2e, config, **kwargs):
		if config.multi_gpu:
			stembedder = nn.DataParallel(stembedder)
			encoder = nn.DataParallel(encoder)
			contextualizer = nn.DataParallel(contextualizer)
			r2e = nn.DataParallel(r2e)
			decoder = nn.DataParallel(decoder)
		if config.encoder_grad_frac < 1:
            # TODO: I hope this works...
			for band, encoder_decoder in encoder.encoder_decoders.items():
				encoder_decoder.patch_embedder.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig 
						  if ig is not None else None for ig in in_grad))
				
				encoder_decoder.gnn_embedder.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig 
						  if ig is not None else None for ig in in_grad))
				
				encoder_decoder.decoders.register_full_backward_hook(lambda module, in_grad, out_grad:
					tuple(config.encoder_grad_frac * ig
						  if ig is not None else None for ig in in_grad))

			'''
			encoder.register_full_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(config.encoder_grad_frac * ig 
												 if ig is not None else None for ig in in_grad))
			'''

		super(MENDRTrainer, self).__init__(encoder=encoder, contextualizer=contextualizer, r2e=r2e, 
			contrastive_loss_fn=nn.CrossEntropyLoss(), lr=config.learning_rate, l2_weight_decay=config.l2_weight_decay,
			metrics=dict(Accuracy=self._contrastive_accuracy), 
			save_model_directory=config.save_model_directory, **kwargs)
		
		# Initialize temperature as a trainable parameter
		self.temp = torch.nn.Parameter(torch.tensor(config.temp, requires_grad=True))
		# TODO: Fix these later...
		self.alpha = config.enc_feat_l2
		self.band_coeffs = {
			'delta': 0,
			'theta': 1,
			'alpha': 1,
			'beta': 0,
			'gamma': 0,
		}

		self.num_negatives = config.num_negatives

	
	def forward(self, data):
		relevant_bands = [data[band].float().to(self.device) for band in BANDS]
		inputs = dict(zip(BANDS, relevant_bands))
		encoder_output = self.encoder(data['graph'], inputs)
		contextualizer_output, shape, wavelet_embeddings = self.contextualizer(encoder_output)
		loss, correct, pairs = self.leave_one_out(wavelet_embeddings, self.contrastive_loss_fn)

		'''
		Convert z and negatives into SPD matrices 
		self.contextualizer.freeze_features(unfreeze=False)
		self.r2e.freeze_features(unfreeze=False)

		spd_z, shape = self.contextualizer(unmasked_z)
		spd_z = self.r2e(spd_z, shape)

		spd_negatives, shape = self.contextualizer(negatives[:, :, 0, :].permute([0, 2, 1]))
		spd_negatives = self.r2e(spd_negatives, shape)

		self.contextualizer.freeze_features(unfreeze=True)
		self.r2e.freeze_features(unfreeze=True)

		# Prediction -> batch_size x predict_length x predict_length
		logits = self._calculate_similarity(z=spd_z, c=c, negatives=spd_negatives.permute([0, 2, 1])[:, :, None, :])
		return logits, z, mask, decoded_coefficients
		'''
		return
	
	def calculate_loss(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
		recon_loss = self._reconstruction_loss(inputs, wavelet_recons)
		return self.loss_fn(logits, labels) + self.alpha * outputs[1].pow(2).mean() + recon_loss, recon_loss

	def _reconstruction_loss(self, input, outputs):
		decodings = {band: outputs[1] for band, outputs in outputs.items()}
		return WaveletReconstructionLoss(inputs, decodings)
				    
	def _contrastive_accuracy(self, inputs, outputs):
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		return self._simple_accuracy([labels], logits)
    
	def calculate_metrics(self, *inputs, outputs, recon_loss):
		"""
		Cosine Similarity from Calculating Similarity
		"""
		# "Logits" from _calculate_similarity
		logits = outputs[0]
		labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
		
		means = logits.mean(dim=0)
		return {
			'Negative Similarity': means[0].item(),
			'Positive Similarity': means[1].item(),
			'Contrastive Accuracy': self._simple_accuracy([labels], logits),
			'MASK_pct': self._mask_pct(inputs, outputs),
			'BENDR Reconstruction MSE': recon_loss.item()
		}
	
	def leave_one_out(self, embeddings, criterion):
		"""
		Compute leave-one-out loss for wavelet embeddings.

		Args:
			embeddings (dict): Dictionary of SPD wavelet embeddings.
			criterion: Loss function (e.g., CrossEntropyLoss).
			temperature (torch.nn.Parameter): Temperature parameter for scaling logits.

		Returns:
			loss (torch.Tensor): Total leave-one-out loss.
			correct (int): Number of correct predictions.
			pairs (int): Number of prediction pairs.
		"""
		frequency_bands = list(embeddings.keys())
		num_targets = len(frequency_bands)
		loss = 0.0
		correct = 0
		pairs = 0

		for i in range(num_targets):
			# Average embeddings of all other modalities
			other_embeddings = []
			for j in list(range(i)) + list(range(i + 1, num_targets)):
				embedding_tensor = embeddings[frequency_bands[j]][0]
				original_batch_shape = embeddings[frequency_bands[j]][1] #TODO: Is this never used?
				other_embeddings.append(embedding_tensor)
			other_embeddings = torch.stack(other_embeddings).sum(0) / (num_targets - 1)

			assert torch.allclose(other_embeddings, other_embeddings.mT, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {other_embeddings}"
			# Compute logits
			logits = self._batchWiseMatrixSimilarity(embeddings[frequency_bands[i]][0], other_embeddings) * torch.exp(self.temp)
			labels = torch.arange(logits.shape[0], device=self.device)
			#labels = self._gen_labels(original_batch_shape)

			# Forward loss
			l = criterion(logits, labels)
			loss += l
			correct += (torch.argmax(logits, axis=0) == labels).sum().item()
			pairs += logits.size(0)

			# Reverse loss
			l = criterion(logits.T, labels)
			loss += l
			correct += (torch.argmax(logits, axis=1) == labels).sum().item()
			pairs += logits.size(0)

		return loss, correct, pairs

	def _gen_labels(self, batch_shape):
		output = []
		for i in range(batch_shape[0]):
			for j in range(batch_shape[1]):
				output.append(i)
		output = torch.tensor(output).to(self.device)
		return output

	def _matrixCosineSimilarity(self, A, B):
		# Ensure matrices are symmetric
		assert torch.allclose(A, A.T, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {A}"
		assert torch.allclose(B, B.T, atol=(10 ** -PRECISION)), f"Input Matrix Not Symmetric, {B}"

		_, a_eigenvec = torch.linalg.eigh(A, UPLO='L')
		_, b_eigenvec = torch.linalg.eigh(B, UPLO='L')

		sim = 0
		assert a_eigenvec.shape == b_eigenvec.shape
		for i in range(a_eigenvec.shape[1]):
			sim += torch.dot(a_eigenvec[:, i], b_eigenvec[:, i])
		return sim

	def _batchWiseMatrixSimilarity(self, batch_A, batch_B):
		assert batch_A.shape == batch_B.shape
		B, N, N = batch_A.shape
		''' This could be accelerated '''
		output = torch.zeros(B, B).to(self.device)

		for i in range(B):
			for j in range(B):
				output[i, j] = self._matrixCosineSimilarity(batch_A[i], batch_B[i])
		return output


	@staticmethod
	def _simple_accuracy(inputs, outputs):
		if isinstance(outputs, (list, tuple)):
			outputs = outputs[0]
        # average over last dimensions
		while len(outputs.shape) >= 3:
			outputs = outputs.mean(dim=-1)
		return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()
	