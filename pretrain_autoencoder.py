import os
import gc
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from Model.MENDR.Autoencoder.MENDREncoder import MENDRPatchEncoder
from Model.MENDR.Autoencoder.MENDRAutoencoderTrainer import MENDRAutoencoderTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetPretrain import WaveletPretrainDataset, WaveletPretrainConcatDataset

@hydra.main(version_base="1.2", 
            config_path="Model/MENDR/Autoencoder/autoencoder_experiment_configs/",
            config_name="default")
def main(cfg:DictConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE",1))
    rank = int(os.environ.get("RANK",0))
    local_rank = int(os.environ.get("LOCAL_RANK",0))

    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Start Run
    print("Job Started. Parameters:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.training_params.ckpt_dir is not None:
        if cfg.training_params.val_frac <= 0:
            raise Exception("Must have validation dataset to save to dir")

    if rank == 0:
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
    dataset = WaveletPretrainDataset(root=cfg.training_params.input_dir, frac=total_frac, include_high="high" in cfg.meta_params.bands)
    if "input_dir_2" in cfg.training_params:
        print(f"Second Data Dir specified: {cfg.training_params.input_dir_2}")
        dataset2 = WaveletPretrainDataset(root=cfg.training_params.input_dir_2, frac=total_frac, include_high="high" in cfg.meta_params.bands)
        dataset = WaveletPretrainConcatDataset([dataset, dataset2])
    print("*" * 50)
    print("Dataset Loaded. Length of Dataset: ", len(dataset), " given frac: ", total_frac)
    print("Num Subjects: ", dataset.num_subjects)

    ### Model ###
    mendr_autoencoder = MENDRPatchEncoder(**cfg.patch_encoder_params, num_subjects=dataset.num_subjects, device=device, include_high="high" in cfg.meta_params.bands)
    if distributed:
        mendr_autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mendr_autoencoder)
        mendr_autoencoder = torch.nn.parallel.DistributedDataParallel(
            mendr_autoencoder.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    else:
        mendr_autoencoder = mendr_autoencoder.to(device)
    optimizer = torch.optim.AdamW(mendr_autoencoder.parameters(),
                betas=(0.9, 0.99),
                lr=cfg.training_params.learning_rate,
                weight_decay=cfg.training_params.l2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=cfg.training_params.T_max,
                eta_min=cfg.training_params.eta_min)
    mix_optimizer = MixOptimizer(optimizer, scheduler)
    trainer = MENDRAutoencoderTrainer(mendr_autoencoder, mix_optimizer, cfg, cuda=device)

    core_model = mendr_autoencoder.module if distributed else mendr_autoencoder

    total_encoder_params = 0
    total_decoder_params = 0
    for band in cfg.meta_params.bands:
        encoder_params = core_model.encoder_decoders[band].getEncoderParamCount()
        decoder_params = core_model.encoder_decoders[band].getDecoderParamCount()
        total_encoder_params += encoder_params
        total_decoder_params += decoder_params
        print(f'{band} Encoder Params: {encoder_params}')
        print(f'{band} Decoder Params: {decoder_params}')
        total_encoder_params += encoder_params
        total_decoder_params += decoder_params
    print("Total Encoder Params: ", total_encoder_params)
    print("Total Decoder Params: ", total_decoder_params)
    print("Total number of parameters: ", total_encoder_params + total_decoder_params)

    ### Training ###
    if cfg.training_params.val_frac > 0:
        print("Splitting Dataset into Train and Validation because Val Fraction > 0.")
        num_train = int(len(dataset) * (cfg.training_params.train_frac / (total_frac)))
        num_val = len(dataset) - num_train
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
        train_sampler = (DistributedSampler(train_dataset,num_replicas=world_size,rank=rank, shuffle=True) if distributed else None)
        val_sampler = (DistributedSampler(val_dataset,num_replicas=world_size, rank=rank, shuffle=False) if distributed else None)
        print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
        trainer.fit(train_dataset, cfg, val_dataset, train_sampler, val_sampler)
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
    
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    print("Cleanup complete.")

if __name__ == '__main__':
    main()