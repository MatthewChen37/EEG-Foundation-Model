import os
import gc
import ast
import copy
import time
from pathlib import Path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

import hydra
from omegaconf import DictConfig, OmegaConf

from Model.MENDR.Autoencoder.MENDRAutoencoderTrainer import MENDRAutoencoderTrainer

@hydra.main(version_base=None, config_path="Model/MENDR/Autoencoder/autoencoder_experiment_configs/", config_name="simple")
def main(cfg : DictConfig) -> None:
    # Start Run
    print(OmegaConf.to_yaml(cfg))
    return None

if __name__ == '__main__':
    main()