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

from math import ceil
from Model.MENDR.MENDR import MENDR_model
from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.Contextualizer.Large.MENDRContextualizerLarge import MENDRContextualizerLarge
from Model.MENDR.Contextualizer.Tiny.MENDRContextualizerTiny import MENDRContextualizerTiny
from Model.MENDR.Downstream.DownstreamDecoders import TUABFinetuneDecoder
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetTUAB import WaveletTUABDataset

@hydra.main(version_base="1.2", 
            config_path="Model/MENDR/Downstream/downstream_experiment_configs/",
            config_name="TUAB")
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

    # Load Dataset
    print("*" * 50)
    if cfg.dataset_params.name == "TUAB":
        finetune_train_dataset = WaveletTUABDataset(root=cfg.dataset_params.train_data_dir, frac=cfg.dataset_params.train_frac)
        finetune_eval_dataset = WaveletTUABDataset(root=cfg.dataset_params.eval_data_dir, frac=cfg.dataset_params.eval_frac)

        model_decoder = TUABFinetuneDecoder().to(device)
    else:
        raise Exception("Dataset not found")
    print("Dataset Loaded. Length of Train Dataset: ", len(finetune_train_dataset))
    print("Dataset Loaded. Length of Validation Dataset: ", len(finetune_eval_dataset))

    ### Model ###
    mendr_autoencoder = MENDRPatchEncoder(**cfg.patch_encoder_params,
                                        device=device)

    for band, encoder_decoder in mendr_autoencoder.encoder_decoders.items():
        encoder_decoder.disableDecoder()

    if cfg.model_params.contextualizer_size == "TINY":
        contextualizer = MENDRContextualizerTiny(
            num_channels=19,
            out_dim=mendr_autoencoder.encoder_decoders['delta'].out_dim,
        ).to(device)
    else:
        raise Exception("Contextualizer size not found")

    if cfg.model_params.pretrained_autoencoder_path is not None:
        mendr_autoencoder.load_state_dict(torch.load(cfg.model_params.pretrained_autoencoder_path, weights_only=True), strict=False)
    if cfg.model_params.pretrained_contextualizer_path is not None:
        contextualizer.load_state_dict(torch.load(cfg.model_params.pretrained_contextualizer_path, weights_only=True), strict=False)

    model = MENDR_model(mendr_autoencoder, contextualizer, device=device, contextualizer_size=cfg.model_params.contextualizer_size).to(device)

    optim_params = []
    optim_params += model.parameters()
    optim_params += list(model_decoder.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=cfg.training_params.epochs*ceil(len(finetune_train_dataset) / cfg.training_params.batch_size),
                                                            eta_min=1e-7)
    optimizer = MixOptimizer(optimizer, scheduler)




if __name__ == '__main__':
    main()