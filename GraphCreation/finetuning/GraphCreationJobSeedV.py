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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch_geometric.data import Data
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix
import pywt

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def main(args):
	files = os.listdir(args.input_directory)
	files = [f for f in files if f.endswith(".fif")]
	files_grouped = group_list(files, 32)
	print(f"Processing {len(files_grouped)} groups")
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_file, args, files_group) for files_group in files_grouped]
		for future in tqdm(futures):
			future.result()

def _process_file(args, files):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_file_thread, args, file) for file in files]
		for future in futures:
			future.result()
    
def _process_file_thread(args, file):
	base_path = os.path.join(args.input_directory, file)
	if os.path.exists(base_path):
		output_dir = os.path.join(args.output_directory, f"{file[:-8]}")
		Path(os.path.join(output_dir, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
		Path(os.path.join(output_dir, "graphs")).mkdir(parents=True, exist_ok=True)
		raw_epoch = mne.read_epochs(base_path, preload=True, verbose=False)
		raw_data = raw_epoch.get_data(copy=True)
		annotations = raw_epoch.get_annotations_per_epoch()
		label = float(annotations[0][0][2])
		dbt = pywt.WaveletPacket(raw_data, wavelet='db4', maxlevel=5, axis=-1)
		relevant_bands = {
			'delta': dbt['aaaaa'].data,
			'theta': dbt['aaaad'].data,
			'alpha': dbt['aaad'].data,
			'beta': dbt['aad'].data,
			'gamma': dbt['ad'].data,
			'high_freq': dbt['d'].data
		}
		for band, data in relevant_bands.items():
			for epoch_idx in range(data.shape[0]):
				torch.save(torch.tensor(data[epoch_idx]), os.path.join(output_dir, "wavelet_decompositions", f"{file[:-4]}_{band}_band_epoch_{epoch_idx}.pt"))

		# TODO: Add support for multiple features
		dist_feat = createDistanceMatrix(raw_epoch.info)
		electrode_pos = createPositionMatrix(raw_epoch.info)
		# Fully connected graph
		edge_indices, edge_weights = createEdges([dist_feat])
		# Create graph
		data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos, y=torch.tensor([label]))
		torch.save(data, os.path.join(output_dir, "graphs", f"{file[:-4]}_graph.pt"))

def parse_args():
	parser = argparse.ArgumentParser(description='Create graphs for TUAB')
	parser.add_argument('--input_directory', type=str, required=True,
					  help='Path to epoched data.')
	parser.add_argument('--output_directory', type=str, required=True,
					  help='Output directory for the graphs.')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	Path(args.output_directory).mkdir(parents=True, exist_ok=True)
	main(args)