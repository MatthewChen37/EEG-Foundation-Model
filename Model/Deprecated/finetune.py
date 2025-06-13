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

from Datasets.datasetTUAB import WaveletTUABDataset
from Datasets.datasetTUEV import WaveletTUEVDataset

@hydra.main(version_base="1.2", 
            config_path="Model/MENDR/Autoencoder/autoencoder_experiment_configs/",
            config_name="sweep_param")
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
	if cfg.dataset_params.name== 'TUAB':
		finetune_train_dataset = WaveletTUABDataset(root=cfg.dataset_params.train_data_dir, frac=cfg.dataset_params.train_frac)
		finetune_eval_dataset = WaveletTUABDataset(root=cfg.dataset_params.val_data_dir, frac=cfg.dataset_params.val_frac)
	elif cfg.dataset_params.dataset == 'TUEV':
		finetune_train_dataset = WaveletTUEVDataset(root=cfg.dataset_params.train_data_dir, frac=cfg.dataset_params.train_frac)
		finetune_eval_dataset = WaveletTUEVDataset(root=cfg.dataset_params.val_data_dir, frac=cfg.dataset_params.val_frac)
	else:
		raise ValueError(f"Unsupported dataset: {cfg.dataset_params.dataset}")