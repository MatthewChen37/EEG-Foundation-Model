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

from Model.MENDR.MENDR import MENDR_model
from Model.MENDR.MENDRPreTrainer import MENDRPreTrainer
from Model.MENDR.mAtt.optimizer import MixOptimizer
from Datasets.datasetPretrain import WaveletPretrainDataset

def main(args):
	# Start Run
	print("Job Started. Parameters:")
	print(" \n".join(f"{k}={v}" for k, v in vars(args).items()))
	print(" \n".join(f"type({k})={type(v)}" for k, v in vars(args).items()))

	BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

	# Create input directory if it doesn't exist
	if args.ckpt_dir is not None:
		if args.val_frac <= 0:
			raise Exception("Must have validation dataset to save to dir")

	if args.model_size == "LARGE" and args.negatives_loo == None:
		raise Exception("Must specify number of negatives for LARGE model")
	
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
	dataset = WaveletPretrainDataset(root=args.input_dir, frac=args.train_frac + args.val_frac)
	if args.input_dir_2:
		print(f"Second Data Dir specified: {args.input_dir_2}")
		dataset2 = WaveletPretrainDataset(root=args.input_dir_2, frac=args.train_frac + args.val_frac)
		dataset = torchdata.ConcatDataset([dataset, dataset2])
	print("*" * 50)
	print("Dataset Loaded. Length of Dataset: ", len(dataset), " given frac: ", args.train_frac + args.val_frac)

	### Model ###
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("Starting training.")

	### Training ###
	mendr = MENDR_model(device, contextualizer_size=args.model_size, n_gnn_transformer_layers=args.n_gnn_transformer_layers)
	trainer = MENDRPreTrainer(mendr, args)
	optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.learning_rate, weight_decay=args.l2_weight_decay)
	optimizer = MixOptimizer(optimizer)
	trainer.set_optimizer(optimizer)

	total_encoder_params = 0
	total_decoder_params = 0
	for band in BANDS:
		print(f'{band} Encoder Params: {mendr.mendr_encoder.encoder_decoders[band].getEncoderParamCount()}')
		print(f'{band} Decoder Params: {mendr.mendr_encoder.encoder_decoders[band].getDecoderParamCount()}')
		total_encoder_params += mendr.mendr_encoder.encoder_decoders[band].getEncoderParamCount()
		total_decoder_params += mendr.mendr_encoder.encoder_decoders[band].getDecoderParamCount()
	if args.model_size.upper() == 'TINY':
		print(f'Wavelet Contextualizer Params: {sum(p.numel() for p in mendr.mendr_contextualizer.Contextualizer.parameters() if p.requires_grad)}')
	elif args.model_size.upper() == 'LARGE':
		print(f'Wavelet Contextualizer Params: {sum(p.numel() for p in mendr.mendr_contextualizer.WaveletContextualizer.parameters() if p.requires_grad)}')
		print(f'Combined Contextualizer Params: {sum(p.numel() for p in mendr.mendr_contextualizer.CombinedContextualizer.parameters() if p.requires_grad)}')
	else:
		raise ValueError("Unidentified Contextualizer Type")

	print("Total Encoder Params: ", total_encoder_params)
	print("Total Decoder Params: ", total_decoder_params)
	print("Total number of parameters: ", sum(p.numel() for p in trainer.parameters() if p.requires_grad))
	
	if args.load_from_ckpt:
		print(f'Checkpoint specified. Loading from checkpoint: {args.load_from_ckpt}')
		if not os.path.exists(args.load_from_ckpt):
			raise Exception(f"Checkpoint folder {args.load_from_ckpt} does not exist.")
		assert set(os.listdir(args.load_from_ckpt)) == {'mendr_model_weights.pth', 'scheduler.pth'}
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
		"--mask_ratio", type=float, help="Ratio of patches to mask", default=0.5
	)

	parser.add_argument(
		"--negatives_loo", type=int, help="Number of negatives to use for loss", default=20
	)

	parser.add_argument(
		"--model_size", type=str, help="Model size", default="LARGE"
	)	

	parser.add_argument(
		"--ckpt_dir", type=str, help="If not none, save after each epoch if validation loss decreases.", required=True, default=None
	)

	parser.add_argument(
		"--load_from_ckpt", type=str, help="Initializes weights of encoder and contextualizer with weights from folder. If it cannot throws an error.", required=False, default=None
	)

	parser.add_argument(
		"--contrastive_loss_pref", type=float, help="Weight of  contrastive loss", default=1.0
	)

	parser.add_argument(
		"--contrastive_combined_loss_coeff", type=float, help="Coefficient of combined contrastive loss", default=1.0
	)

	parser.add_argument(
		"--contrastive_wavelet_loss_coeff", type=float, help="Coefficient of wavelet contrastive loss", default=1e3
	)

	parser.add_argument(
		"--delta_reconstructive_loss_pref", type=float, help="Weight of delta reconstructive loss in Jacobian Descent", default=1.0
	)

	parser.add_argument(
		"--theta_reconstructive_loss_pref", type=float, help="Weight of theta reconstructive loss in Jacobian Descent", default=1.0
	)

	parser.add_argument(
		"--alpha_reconstructive_loss_pref", type=float, help="Weight of alpha reconstructive loss in Jacobian Descent", default=1.0
	)

	parser.add_argument(
		"--beta_reconstructive_loss_pref", type=float, help="Weight of beta reconstructive loss in Jacobian Descent", default=1.0
	)

	parser.add_argument(
		"--gamma_reconstructive_loss_pref", type=float, help="Weight of gamma reconstructive loss in Jacobian Descent", default=1.0
	)

	parser.add_argument(
		"--gradient_clip_value", type=float, help="Gradient Clipping Value", default=1e7
	)

	'''
	sum: Weighted sum of losses (default)
	Here contrastive_combined_loss_coeff, contrastive_wavelet_loss_coeff, delta_reconstructive_loss_pref, theta_reconstructive_loss_pref, alpha_reconstructive_loss_pref, beta_reconstructive_loss_pref, gamma_reconstructive_loss_pref are used as coefficients for each loss

	real_time: As mentioned in this article: https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0
	Scaling all losses to 1, and then adding them up

	jacobian: TorchJD implementation of the jacobian descent strategy
	Here contrastive_combined_loss_coeff is used as coefficient for the combined loss and contrastive_wavelet_loss_coeff is used as coefficient for the wavelet loss, which are summed into one loss

	the contrastive_loss_pref is used as coefficient magnitude of projection 
	the delta_reconstructive_loss_pref, theta_reconstructive_loss_pref, alpha_reconstructive_loss_pref, beta_reconstructive_loss_pref, gamma_reconstructive_loss_pref are used as coefficient magnitude of projection
	'''
	parser.add_argument(
		"--multi_objective_loss_balancing_strategy",
		type=str,
		help="How to balance losses: https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0, Possible values: sum, real_time, or jacobian",
		default="sum"
	)

	parser.add_argument(
		"--n_gnn_transformer_layers",
		type=int,
		help="Number of GNN Transformer layers",
		default=2
	)

	# parse args
	args = parser.parse_args()
	return args
	
if __name__ == "__main__":
	args = parse_args()
	main(args)