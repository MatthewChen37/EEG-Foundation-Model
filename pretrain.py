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
from Model.transforms import RandomTemporalCrop

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
	SpatialTemporalEncoder = SpatialTemporalEncoder(args)
	encoder = ConvEncoder(in_features=args.d_model, encoder_h=args.encoder_h, 
					   enc_width=args.enc_width, dropout=args.enc_dropout, enc_downsample=args.enc_downsample)
	contextualizer = mATTContextualizer(args)
	r2e = R2E(args)

	### Training ###
	trainer = MENDRTrainer(SpatialTemporalEncoder, encoder, contextualizer, r2e, args)
	trainer.set_optimizer(torch.optim.Adam(trainer.parameters()))
	trainer.add_batch_transform(RandomTemporalCrop(max_crop_frac=args.max_crop_frac))

	# trainer.fit(training_dataset=datasetList, epochs=5, batch_size=args.batch_size)


def parse_args():
	# setup arg parser
	parser = argparse.ArgumentParser()

	# Training Parameters
	parser.add_argument(
		'--input_dir', type=str, help='Path to training data', required=True
	)

	parser.add_argument(
		"--random_state", type=int, help="Random state for reproducibility", default=42
	)
	parser.add_argument(
        "-b", "--batch_size", default=512, type=int, help="mini-batch size (default: 512)"
    )

	parser.add_argument("-e", "--epochs", default=100, type=int, help="number of total epochs (default: 100)",
    )

	# Spatial Temporal Embedding Encoder Configs
	parser.add_argument(
		"--seq_len", type=int, help="Sequence Length of Recordings", required=True, default=15360
	)

	parser.add_argument(
		"--pred_len", type=int, help="Prediction Length of Recordings, should be equal to sequence length", required=True, default=15360
	)

	parser.add_argument(
		"--top_k", type=int, help="Top K frequencies to select in TimesBlock", required=True, default=5
	)

	parser.add_argument(
		"--d_model", type=int, help="Number of EEG Channels", required=True, default=19
	)

	parser.add_argument(
		"--d_ff", type=int, help="Feed Forward Dimension in TimesBlock", required=True, default=9
	)

	parser.add_argument(
		"--num_kernels", type=int, help="Number of Kernels in Convolutional Layer of Inception Block in TimesBlock", required=True, default=3
	)

	parser.add_argument(
		"--num_heads", type=int, help="Number of Heads in Multihead Attention Layer of GAT", required=True, default=4
	)

	parser.add_argument(
		"--dropout", type=float, help="Dropout Rate for GAT", required=True, default=0.1
	)

	# Encoder Configs
	parser.add_argument(
		"--encoder_h", type=int, help="Hidden Dimension of Encoder", required=True, default=256
	)

	parser.add_argument(
		"--enc_width", type=ast.literal_eval, help="Encoder Width", required=True, default=(3, 2, 2)
	)

	parser.add_argument(
		"--enc_downsample", type=ast.literal_eval, help="Encoder Downsample", required=True, default=(3, 2, 2)
	)

	parser.add_argument(
		"--enc_dropout", type=float, help="Dropout Rate for Encoder", required=True, default=0.1
	)

	# MENDR Contextualizer Configs
	parser.add_argument(
		"--in_features", type=int, help="Input Features for Contextualizer", required=True, default=256
	)

	parser.add_argument(
		"--dropout", type=float, help="Dropout Rate for Contextualizer", required=True, default=0.1
	)

	parser.add_argument(
		"--position_encoder", type=int, help="Position Encoder", required=True, default=25
	)

	parser.add_argument(
		"--epochs", type=int, help="Number of Epochs for mATT", required=True, default=4
	)

	# Trainer Configs
	parser.add_argument(
		"--mask_span", type=int, help="Mask Span", required=True, default=6
	)

	parser.add_argument(
		"--multi_gpu", type=bool, help="Multi GPU Training", required=True, default=False
	)

	parser.add_argument(
		"--encoder_grad_frac", type=float, help="Encoder Gradient Fraction", required=True, default=1
	)

	parser.add_argument(
		"--learning_rate", type=float, help="Learning Rate", required=True, default=1e-3
	)

	parser.add_argument(
		"--l2_weight_decay", type=float, help="L2 Weight Decay", required=True, default=1e-5
	)

	parser.add_argument(
		"--mask_rate", type=float, help="Mask Rate", required=True, default=0.1
	)

	parser.add_argument(
		"--temp", type=float, help="Temperature", required=True, default=0.1
	)

	parser.add_argument(
		"--permuted_encodings", type=bool, help="Permuted Encodings", required=True, default=False
	)

	parser.add_argument(
		"--permuted_contexts", type=bool, help="Permuted Contexts", required=True, default=False
	)

	parser.add_argument(
		"--enc_feat_l2", type=float, help="Encoder Feature L2", required=True, default=1e-5
	)

	parser.add_argument(
		"--unmasked_negative_frac", type=float, help="Unmasked Negative Fraction", required=True, default=0.1
	)

	parser.add_argument(
		"--num_negatives", type=int, help="Number of Negatives in Contrastive Learning Task", required=True, default=10
	)

	parser.add_argument(
		"--max_crop_frac", type=float, help="Maximum Crop Fraction on Random Temporal Crop", required=True, default=0.05
	)

	# parse args
	args = parser.parse_args()
	return args
	

if __name__ == "__main__":
	args = parse_args()
	main(args)