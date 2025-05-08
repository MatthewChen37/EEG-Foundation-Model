import mne, torch
import sys
sys.path.append("../")
import os, argparse
import numpy as np
import pandas as pd
import warnings
import traceback
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pywt

def main(args):
	files = None # TODO



def parse_args():
	parser = argparse.ArgumentParser(description="Process PhysioNet data")
	parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the PhysioNet data")
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
	return parser.parse_args()

if __name__ == "__main__":
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)