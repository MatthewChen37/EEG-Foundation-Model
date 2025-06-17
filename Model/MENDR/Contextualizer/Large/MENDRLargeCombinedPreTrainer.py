import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.baseModelTrainer import BaseModelTrainer
from Model.MENDR.safeSVD import SVD, svdv2
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge
from Explainability.embeddingVisualization import plotSPDEmbedding
from Explainability.plotWaveletEmbeddings import plotWaveletEmbeddingsRiemannian
import mlflow
import matplotlib.pyplot as plt
import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
class MENDRLargeCombinedPreTrainer(BaseModelTrainer):
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

		assert isinstance(contextualizer, MENDRContextualizerLarge), f"Contextualizer must be of type MENDRContextualizerLarge, but got {type(contextualizer)}"

		super(MENDRLargeCombinedPreTrainer, self).__init__(autoencoder=autoencoder, contextualizer=contextualizer,
			contrastive_loss_fn_wavelet=self.contrastive_loss_fn_wavelet,
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

	def forward(self, data):
		if self.train_mode == "wavelet":
			patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)
			w_loss, w_correct, w_pairs = self.leave_one_out(wavelet_manifold_output, self.contrastive_loss_fn_wavelet, negatives=self.negatives_loo)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'wavelet_manifold_output': wavelet_manifold_output,
					'wavelet_loss': w_loss,
					'wavelet_acc': w_correct / w_pairs,
					}
		elif self.train_mode == "combined":
			patchified_inputs, encodings, decodings = self.autoencoder.forward(data)
			wavelet_manifold_output, epoched_shape = self.mendr_model.mendr_contextualizer.WaveletContextualizer(encodings)
			riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes = self.epochMaskedRecon(wavelet_manifold_output,
																															epoched_shape,
																															self.contrastive_loss_fn_combined)
			return {
					'patchified_inputs': patchified_inputs,
					'encodings': encodings,
					'decodings': decodings,
					'combined_manifold_output': combined_manifold_output,
					'wavelet_manifold_output': wavelet_manifold_output,
				}
		else:
			raise ValueError(f"Invalid training mode: {self.train_mode}. Must be 'wavelet' or 'combined'.")
		
	def epochMaskedRecon(self, wavelet_manifold_output, epoched_shape, criterion):
		frequency_bands = list(wavelet_manifold_output.keys())
		batch_size = epoched_shape[0]
		num_epochs = epoched_shape[1]
		# Only ever have a non-zero mask ratio HERE
		combined_manifold_output = self.mendr_model.mendr_contextualizer.CombinedContextualizer._wavelet_LogEuclideanMean(wavelet_manifold_output)
		combined_manifold_output_masked = combined_manifold_output.clone()
		combined_manifold_output_masked, mask_idxes = self.mendr_model.mendr_contextualizer.CombinedContextualizer(combined_manifold_output_masked, epoched_shape, mask_ratio=self.mask_ratio)
		combined_manifold_output = combined_manifold_output.view(epoched_shape[0], epoched_shape[1], combined_manifold_output.shape[2], combined_manifold_output.shape[3])

		# Masked Reconstruction loss
		# Only compare loss of masked parts
		riemannian_loss = criterion(combined_manifold_output[mask_idxes], combined_manifold_output_masked[mask_idxes])
		return self.contrastive_combined_loss_coeff * riemannian_loss, combined_manifold_output, combined_manifold_output_masked, mask_idxes



