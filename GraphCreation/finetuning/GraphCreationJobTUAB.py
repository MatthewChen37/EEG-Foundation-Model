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

failed_files = []
def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def main(args):
	classes = os.listdir(args.input_directory)
	assert len(classes) == 2, "Only two classes should be found, normal and abnormal"
	
	# Define the columns for the empty DataFrame
	columns = ['f_path', 'error', 'traceback'] 

	for c in classes:
		subclass_folder = os.path.join(args.input_directory, c)
		files = os.listdir(subclass_folder)
		files_grouped = group_list(files, 160)
		print(f"Processing {len(files_grouped)} groups for class {c}")
		with ProcessPoolExecutor() as executor:
			futures = [executor.submit(_process_file, args, c, files_group) for files_group in files_grouped]
			for future in tqdm(futures):
				future.result()
	failed_files_df = pd.DataFrame(failed_files, columns=columns)
	failed_files_df.to_csv(os.path.join(args.output_directory, "failed_files.csv"), index=False)

def _process_file(args, c, files):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_file_thread, args, c, file) for file in files]
		for future in futures:
			result = future.result()
			if result is not None:
				failed_files.append(result)
	return None

    
def _process_file_thread(args, c, file):
	base_path = os.path.join(args.input_directory, c, file)
	if os.path.exists(base_path):
		try:
			subject = file.split("/")[-1]
			output_dir = os.path.join(args.output_directory, f"{subject}-{c}")
			Path(os.path.join(output_dir, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(output_dir, "graphs_v2")).mkdir(parents=True, exist_ok=True)
			raw_epoch = mne.read_epochs(base_path, preload=True, verbose=False)
			raw_data = raw_epoch.get_data(copy=True)
			dbt = pywt.WaveletPacket(raw_data, wavelet='db4', maxlevel=5, axis=-1)
			relevant_bands = {
				'delta': dbt['aaaaa'],
				'theta': dbt['aaaad'],
				'alpha': dbt['aaad'],
				'beta': dbt['aad'],
				'gamma': dbt['ad'],
				'high_freq': dbt['d']
			}
			for band, data in relevant_bands.items():
				torch.save(data, os.path.join(output_dir, "wavelet_decompositions", f"{file[:-4]}_{band}_band.pt"))
			# TODO: Add support for multiple features
			dist_feat = createDistanceMatrix(raw_epoch.info)
			electrode_pos = createPositionMatrix(raw_epoch.info)
			# Fully connected graph
			edge_indices, edge_weights = createEdges([dist_feat])
			# Create graph
			data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
			torch.save(data, os.path.join(output_dir, "graphs_v2", f"{file[:-4]}_graph.pt"))
			return None
		except Exception as e:
			return (file, e, traceback.format_exc())
	else:
		return None

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