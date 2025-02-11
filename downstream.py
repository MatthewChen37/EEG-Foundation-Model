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

from dataset import WaveletFinetuningDataset

def main(args):
    # Start Run
    print("Job Started. Parameters:")
    print(" \n".join(f"{k}={v}" for k, v in vars(args).items()))
    print(" \n".join(f"type({k})={type(v)}" for k, v in vars(args).items()))


    ### Seed ###
    torch.cuda.empty_cache()
    random.seed(args.random_state)
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)
    ## CUDNN ##
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load Dataset
    print("*" * 50)
    train_dataset = WaveletFinetuningDataset(root="/home/hice1/mchen439/scratch/downstreamTaskData/train")
    print("Dataset Loaded. Length of Train Dataset: ", len(train_dataset))
    eval_dataset = WaveletFinetuningDataset(root="/home/hice1/mchen439/scratch/downstreamTaskData/eval")
    print("Dataset Loaded. Length of Validation Dataset: ", len(eval_dataset))

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

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_state", type=int, help="Random state for reproducibility", default=42)

    parser.add_argument("--multi_gpu", type=bool, help="Multi GPU Training", default=False)

    # parse args
    args = parser.parse_args()
    return args
	
if __name__ == "__main__":
    args = parse_args()
    main(args)