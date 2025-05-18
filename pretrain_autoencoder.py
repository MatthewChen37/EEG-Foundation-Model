import os
import gc
import ast
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

import hydra
from omegaconf import DictConfig, OmegaConf

from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.Autoencoder.MENDRAutoencoderTrainer import MENDRAutoencoderTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetPretrain import WaveletPretrainDataset


@hydra.main(version_base=None, config_path="Model/MENDR/Autoencoder/autoencoder_experiment_configs/", config_name="default")
def main(cfg:DictConfig) -> None:
    # Start Run
    print("Job Started. Parameters:")
    print(OmegaConf.to_yaml(cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    '''
    # Load Dataset
    total_frac = cfg.training_params.train_frac + cfg.training_params.val_frac
    dataset = WaveletPretrainDataset(root=cfg.training_params.input_dir, frac=total_frac)
    if "input_dir_2" in cfg.training_params:
        print(f"Second Data Dir specified: {cfg.training_params.input_dir_2}")
        dataset2 = WaveletPretrainDataset(root=cfg.training_params.input_dir_2, frac=total_frac)
        dataset = torchdata.ConcatDataset([dataset, dataset2])
    print("*" * 50)
    print("Dataset Loaded. Length of Dataset: ", len(dataset), " given frac: ", total_frac)
    '''

    ### Model ###
    mendr_autoencoder = MENDRPatchEncoder(**cfg.patch_encoder_params, device=device)
    optimizer = torch.optim.AdamW(mendr_autoencoder.parameters(), 
                lr=cfg.training_params.learning_rate,
                weight_decay=cfg.training_params.l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=cfg.training_params.T_max,
                eta_min=cfg.training_params.eta_min)
    mix_optimizer = MixOptimizer(optimizer, scheduler)
    trainer = MENDRAutoencoderTrainer(mendr_autoencoder, mix_optimizer, cfg, cuda=device)
    ### Training ###







    return None

if __name__ == '__main__':
    main()