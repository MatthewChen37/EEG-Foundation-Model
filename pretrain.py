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

from Model.MENDR.MENDREncoder import MENDREncoder, WaveletEncoderDecoder
from Model.MENDR.MENDRContextualizer import MENDRContextualizer
from Model.MENDR.MENDRTrainer import MENDRTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer

from dataset import WaveletDataset

def main(args):
	# Start Run
	print("Job Started. Parameters:")
	print(" \n".join(f"{k}={v}" for k, v in vars(args).items()))
	print(" \n".join(f"type({k})={type(v)}" for k, v in vars(args).items()))


	# Create input directory if it doesn't exist
	if args.ckpt_dir is not None:
		if args.val_frac <= 0:
			raise Exception("Must have validation dataset to save to dir")
		Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

	
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

	if args.train_frac + args.val_frac > 1:
		raise ValueError("Train and Val Fraction should not exceed 1.")
	# Load Dataset
	dataset = WaveletDataset(root=args.input_dir, frac=args.train_frac + args.val_frac)
	if args.input_dir_2:
		print(f"Second Data Dir specified: {args.input_dir_2}")
		dataset2 = WaveletDataset(root=args.input_dir_2, frac=args.train_frac + args.val_frac)
		dataset = torchdata.ConcatDataset([dataset, dataset2])

	print("*" * 50)
	print("Dataset Loaded. Length of Dataset: ", len(dataset), " given frac: ", args.train_frac + args.val_frac)

	### Model ###
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	encoder = MENDREncoder(device=device)
	contextualizer = MENDRContextualizer(device=device)

	print("Starting training.")
	### Training ###
	trainer = MENDRTrainer(encoder, contextualizer, args)

	for band, encoder in encoder.encoder_decoders.items():
		print(f"{band} Encoder Number of Params: {encoder.getEncoderParamCount()}")
		print(f"{band} Decoder Number of Params: {encoder.getDecoderParamCount()}")
	print("Contextualizer parameters: ", sum(p.numel() for p in contextualizer.parameters() if p.requires_grad))
	print("Total number of parameters: ", sum(p.numel() for p in trainer.parameters() if p.requires_grad))
	optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.learning_rate, weight_decay=5e-4)
	optimizer = MixOptimizer(optimizer)
	trainer.set_optimizer(optimizer)

	if args.load_from_ckpt:
		print(f'Checkpoint specified. Loading from checkpoint: {args.load_from_ckpt}')
		if not os.path.exists(args.load_from_ckpt):
			raise Exception(f"Checkpoint folder {args.load_from_ckpt} does not exist.")
		assert set(os.listdir(args.load_from_ckpt)) == {'encoder_weights.pth', 'contextualizer_weights.pth'}
		trainer.load_from_ckpt(args.load_from_ckpt)
		print(f'Weights successfully loaded.')

	# Split Dataset
	if args.val_frac > 0: # Pre-Pretraining Phase
		print("Splitting Dataset into Train and Validation because Val Fraction > 0.")
		num_train = int(len(dataset) * (args.train_frac / (args.train_frac + args.val_frac)))
		num_val = len(dataset) - num_train
		train_dataset, val_dataset = torchdata.random_split(dataset, [num_train, num_val])
		print("Train and Validation Dataset Length: ", len(train_dataset), len(val_dataset))
		trainer.fit(training_dataset=train_dataset, validation_dataset=val_dataset, epochs=args.training_epochs, batch_size=args.batch_size)
	else:
		print("No Validation Set. Training on Whole Dataset.")
		trainer.fit(training_dataset=dataset, epochs=args.training_epochs, batch_size=args.batch_size)

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

	# Training Parameters
	parser.add_argument(
		'--input_dir', type=str, help='Path to training data', required=True
	)

	parser.add_argument(
		'--input_dir_2', type=str, help='Path to more training data (datasets will be concatenated)', required=False
	)

	parser.add_argument(
		"--random_state", type=int, help="Random state for reproducibility", default=42
	)
	parser.add_argument(
        "-b", "--batch_size", default=32, type=int, help="mini-batch size (default: 32)"
    )

	parser.add_argument(
		"-e", "--training_epochs", default=1, type=int, help="number of total training epochs (default: 1)"
	)

	parser.add_argument(
		"--train_frac", type=float, help="Fraction of dataset to use for training", default=1.0
	)

	parser.add_argument(
		"--val_frac", type=float, help="Fraction of dataset to use for validation", default=0.0
	)

	# MENDR Contextualizer Configs
	parser.add_argument(
		"--in_features", type=int, help="Input Features for Contextualizer", default=32
	)

	parser.add_argument(
		"--dropout", type=float, help="Dropout Rate for Contextualizer", default=0.1
	)

	parser.add_argument(
		"--epochs", type=int, help="Number of Epochs for mATT", default=4
	)

	# Trainer Configs
	parser.add_argument(
		"--learning_rate", type=float, help="Learning Rate", default=0.01
	)

	parser.add_argument(
		"--l2_weight_decay", type=float, help="L2 Weight Decay. Helps with generalization", default=1e-5
	)

	parser.add_argument(
		"--temp", type=float, help="Temperature", default=10.0
	)

	parser.add_argument(
		"--enc_feat_l2", type=float, help="Encoder Feature L2", default=1e-5
	)

	parser.add_argument(
		"--ckpt_dir", type=str, help="If not none, save after each epoch if validation loss decreases.", required=True, default=None
	)

	parser.add_argument(
		"--load_from_ckpt", type=str, help="Initializes weights of encoder and contextualizer with weights from folder. If it cannot throws an error.", required=False, default=None
	)

	# parse args
	args = parser.parse_args()
	return args
	
if __name__ == "__main__":
	args = parse_args()
	main(args)