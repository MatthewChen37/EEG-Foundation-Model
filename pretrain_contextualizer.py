import os
import gc
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder

from Model.MENDR.Contextualizer.Tiny.MENDRTinyPreTrainer import MENDRTinyPreTrainer
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetPretrain import WaveletPretrainDataset, WaveletPretrainConcatDataset

@hydra.main(version_base="1.2", 
            config_path="Model/MENDR/Contextualizer/Tiny/tiny_experiment_configs/",
            config_name="default")
def main(cfg:DictConfig) -> None:
	# Start Run
	print("Job Started. Parameters:")
	print(OmegaConf.to_yaml(cfg))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create input directory if it doesn't exist
	if cfg.training_params.ckpt_dir is not None:
		if cfg.training_params.val_frac <= 0:
			raise Exception("Must have validation dataset to save to dir")

	Path(cfg.training_params.ckpt_dir).mkdir(parents=True, exist_ok=True)

	### Seed ###
	torch.cuda.empty_cache()
	random.seed(cfg.training_params.random_seed)
	os.environ['PYTHONHASHSEED'] = str(cfg.training_params.random_seed)
	np.random.seed(cfg.training_params.random_seed)
	torch.manual_seed(cfg.training_params.random_seed)
	torch.cuda.manual_seed(cfg.training_params.random_seed)
	torch.cuda.manual_seed_all(cfg.training_params.random_seed)
	## CUDNN ##
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	# Load Dataset
	total_frac = cfg.training_params.train_frac + cfg.training_params.val_frac
	dataset = WaveletPretrainDataset(root=cfg.training_params.input_dir, frac=total_frac)
	if "input_dir_2" in cfg.training_params:
		print(f"Second Data Dir specified: {cfg.training_params.input_dir_2}")
		dataset2 = WaveletPretrainDataset(root=cfg.training_params.input_dir_2, frac=total_frac)
		dataset = WaveletPretrainConcatDataset([dataset, dataset2])
	print("*" * 50)
	print("Dataset Loaded. Length of Dataset: ", len(dataset), " given frac: ", total_frac)

	### Model ###
	autoencoder = MENDRPatchEncoder(**cfg.patch_encoder_params, num_subjects=dataset.num_subjects, device=device)
	for band, encoder_decoder in autoencoder.encoder_decoders.items():
		encoder_decoder.disableDecoder()
	autoencoder.load_state_dict(torch.load(cfg.training_params.autoencoder_ckpt_path, weights_only=True), strict=False)
	for param in autoencoder.parameters():
		param.requires_grad = False # Freeze the autoencoder
	autoencoder.eval()

	if cfg.meta_params.model_size.lower() == 'tiny':
		contextualizer = MENDRContextualizerTiny(
			encoded_h = cfg.patch_encoder_params.delta_encoded_h +
						cfg.patch_encoder_params.theta_encoded_h +
						cfg.patch_encoder_params.alpha_encoded_h +
						cfg.patch_encoder_params.beta_encoded_h  +
						cfg.patch_encoder_params.gamma_encoded_h,
			patch_len = autoencoder.encoder_decoders['delta'].L_out_2,
			device=device, **cfg.contextualizer_params)
	elif cfg.meta_params.model_size.lower() == 'large':
		raise ValueError("Large contextualizer not supported in this script")

	optimizer = torch.optim.AdamW(contextualizer.parameters(),
				lr=cfg.training_params.learning_rate,
				weight_decay=cfg.training_params.l2_weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
				T_max=cfg.training_params.T_max,
				eta_min=cfg.training_params.eta_min)
	mix_optimizer = MixOptimizer(optimizer, scheduler)
	print(f'Contextualizer Params: {sum(p.numel() for p in contextualizer.parameters() if p.requires_grad)}')

	if cfg.meta_params.model_size.lower() == 'tiny':
		trainer = MENDRTinyPreTrainer(autoencoder, contextualizer, mix_optimizer, cfg, cuda=device)
	else:
		raise ValueError("Large contextualizer not supported in this script")

	### Training ###
	if cfg.training_params.val_frac > 0:
		print("Splitting Dataset into Train and Validation because Val Fraction > 0.")
		num_train = int(len(dataset) * (cfg.training_params.train_frac / (total_frac)))
		num_val = len(dataset) - num_train
		train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
		print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
		trainer.fit(training_dataset=train_dataset, cfg=cfg, validation_dataset=val_dataset)
	else:
		print("No Validation Set. Training on Whole Dataset.")
		trainer.fit(training_dataset=dataset, cfg=cfg)

	
	print("*" * 50)
	print("Cleaning up resources...")
	# Clear the PyTorch cache (for GPU)
	torch.cuda.empty_cache()
	# Force garbage collection (for CPU and GPU tensors)
	gc.collect()
	# If using multiple GPUs, synchronize them (optional but recommended)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	print("Cleanup complete.")

if __name__ == '__main__':
    main()