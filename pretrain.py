import os
import gc
import ast
import copy
import time
import pathlib
import argparse
import random
import numpy as np
import pandas as pd

#import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata



from Model.MENDR.SpatialTemporalEncoder import SpatialTemporalEncoder
from Model.MENDR.MENDRContextualizer import mATTContextualizer
from Model.MENDR.MENDRTrainer import MENDRTrainer
from Model.encoder import ConvEncoder
from Model.MENDR.R2E import R2E

from dataset import EEGDataset

def main(args):
	# Load Dataset
	#dataset = EEGDataset(root=args.input_dir)

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

	### Model ###

	

	# Train Model


def parse_args():
	# setup arg parser
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--input_dir', type=str, help='Path to training data', required=True
	)

	parser.add_argument(
		"--random_state", type=int, default=42, help="Random state for reproducibility"
	)

	# parse args
	args = parser.parse_args()
	return args
	

if __name__ == "__main__":
	args = parse_args()
	main(args)