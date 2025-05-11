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
	splits = os.listdir(args.input_dir)
	print(f"Splits: {splits}")

	for split in splits:
		input_split_dir = os.path.join(args.input_dir, split)
		output_split_dir = os.path.join(args.output_dir, split)
		Path(output_split_dir).mkdir(parents=True, exist_ok=True)
		subjects = os.listdir(input_split_dir)
		with ProcessPoolExecutor() as executor:
			futures = [executor.submit(process_subject, input_split_dir, output_split_dir, subject) for subject in subjects]
			for future in tqdm(futures):
				future.result()

def process_subject(input_split_dir, output_split_dir, subject):
	subject_path = os.path.join(input_split_dir, subject)
	if os.path.isdir(subject_path):
		files = os.listdir(subject_path)
		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(process_file, subject_path, output_split_dir, file) for file in files]
			for future in futures:
				future.result()
	else:
		print(f"Skipping {subject}, not a directory")

def process_file(subject_path, output_split_dir, file):
	base_path = os.path.join(subject_path, file)
	if os.path.exists(base_path):
		output_dir = os.path.join(output_split_dir, file)
		epoch_idx = int(file[:-4].split("_")[-2])
		label = int(file[:-4].split("_")[-1])
		Path(os.path.join(output_dir, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
		Path(os.path.join(output_dir, "graphs")).mkdir(parents=True, exist_ok=True)
		raw = mne.io.read_raw_fif(base_path, preload=True, verbose=False)
		raw_data = raw.get_data()

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
			curr_epoch = torch.tensor(data)
			torch.save(curr_epoch, os.path.join(output_dir, "wavelet_decompositions", f"{file[:-4]}_{band}_band_epoch_{epoch_idx}.pt"))

		dist_feat = createDistanceMatrix(raw.info)
		electrode_pos = createPositionMatrix(raw.info)
		edge_indices, edge_weights = createEdges([dist_feat])

		data_graph = Data(edge_index=edge_indices, pos=electrode_pos, y = torch.tensor([int(label)]), edge_attr=edge_weights)
		torch.save(data_graph, os.path.join(output_dir, "graphs", f"{file[:-4]}_epoch_{epoch_idx}.pt"))
	else:
		print(f"File {base_path} does not exist")

	return None

def parse_args():
	parser = argparse.ArgumentParser(description="Process ISRUC data")
	parser.add_argument("--input_dir", type=str, help="Input directory containing the ISRUC dataset", default="/home/azureuser/mycontainer/ISRUC_Processed/epoched_data")
	parser.add_argument("--output_dir", type=str, help="Output directory for processed data", default="/home/azureuser/mycontainer/ISRUC_Processed/wavelet_data")
	return parser.parse_args()

if __name__ == "__main__":
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)