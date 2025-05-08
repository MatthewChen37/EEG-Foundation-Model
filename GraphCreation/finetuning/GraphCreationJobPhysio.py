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
	files = os.listdir(args.input_dir)
	print(f"Files: {files}")

	for split in files:
		input_split_dir = os.path.join(args.input_dir, split)
		output_split_dir = os.path.join(args.output_dir, split)
		Path(output_split_dir).mkdir(parents=True, exist_ok=True)
		subjects = os.listdir(input_split_dir)
		print(f"Processing {split} with {len(subjects)} subjects")
		with ProcessPoolExecutor() as executor:
			futures = [executor.submit(process_subject, input_split_dir, output_split_dir, subject) for subject in subjects]
			for future in tqdm(futures):
				future.result()
						
def process_subject(input_split_dir, output_split_dir, subject):
	subject_path = os.path.join(input_split_dir, subject)
	if os.path.isdir(subject_path):
		files = os.listdir(subject_path)
		'''
		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(process_file, subject_path, output_split_dir, file) for file in files]
			for future in futures:
				future.result()
		'''
	else:
		print(f"Skipping {subject}, not a directory")

def process_file(subject_path, output_split_dir, file):
	base_path = os.path.join(subject_path, file)
	if os.path.exists(base_path):
		try:
			output_dir = os.path.join(output_split_dir, file)
			Path(os.path.join(output_dir, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(output_dir, "graphs")).mkdir(parents=True, exist_ok=True)
			raw = mne.io.read_raw_fif(base_path, preload=True, verbose=False)
			raw_data = raw.get_data()
			'''
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

			dist_feat = createDistanceMatrix(raw_epoch.info)
			electrode_pos = createPositionMatrix(raw_epoch.info)

			for epoch_idx in range(raw_data.shape[0]):
				data_graph = Data(x=torch.tensor(raw_data[epoch_idx]), edge_index=createEdges(dist_feat), pos=torch.tensor(electrode_pos))
				torch.save(data_graph, os.path.join(output_dir, "graphs", f"{file[:-4]}_epoch_{epoch_idx}.pt"))
			'''
		except Exception as e:
			print(f"Error processing {file}: {e} Base Path: {base_path}")
			print(f"Subject Path: {subject_path} File: {file}")
			traceback.print_exc()

def parse_args():
	parser = argparse.ArgumentParser(description='Create graphs for Physio data.')
	parser.add_argument('--input_dir', type=str, required=True, help='Path to epoched data.', )
	parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the graphs.')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)