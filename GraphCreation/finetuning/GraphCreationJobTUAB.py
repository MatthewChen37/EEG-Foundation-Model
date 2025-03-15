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
		files_grouped = group_list(files, 30)
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
			output_dir = os.path.join(args.output_directory, f"{subject}_{c}")
			wavelet_path = os.path.join(output_dir, "wavelet_decompositions")
			Path(wavelet_path).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(output_dir, "graphs")).mkdir(parents=True, exist_ok=True)
			raw_epoch = mne.read_epochs(base_path, preload=True, verbose=False)
			raw_data = raw_epoch.get_data(copy=True, verbose=False)

			if args.sampling_rate == 128:
				dbt = pywt.WaveletPacket(raw_data, wavelet='db4', maxlevel=5, axis=-1)
				relevant_bands = {
					'delta': dbt['aaaaa'].data,
					'theta': dbt['aaaad'].data,
					'alpha': dbt['aaad'].data,
					'beta': dbt['aad'].data,
					'gamma': dbt['ad'].data,
					'high_freq': dbt['d'].data
				}

			elif args.sampling_rate == 256:
				dbt = pywt.WaveletPacket(raw_data, wavelet='db4', maxlevel=6, axis=-1)
				relevant_bands = {
					'delta': dbt['aaaaaa'].data, # 0 - 4 Hz
					'theta': dbt['aaaaad'].data, # 4 - 8 Hz
					'alpha': dbt['aaaad'].data, # 8 - 16 Hz
					'beta': dbt['aaad'].data, # 16 - 32 Hz
					'gamma': dbt['aad'].data, # 32 - 64 Hz
					'high_freq': dbt['ad'].data # 64 - 128 Hz
                }

			for band, data in relevant_bands.items():
				epoch_data = torch.tensor(data)
				for i in range(epoch_data.shape[0]):
					curr_epoch = torch.tensor(epoch_data[i])
					torch.save(curr_epoch, os.path.join(wavelet_path, f"{file[:-4]}_{band}_band_epoch_{10 + i}.pt"))

			# TODO: Add support for multiple features
			dist_feat = createDistanceMatrix(raw_epoch.info)
			electrode_pos = createPositionMatrix(raw_epoch.info)

			# Fully connected graph
			edge_indices, edge_weights = createEdges([dist_feat])

			# Create graph
			data = Data(x=raw_data, edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
			torch.save(data, os.path.join(output_dir, "graphs", f"{file[:-4]}_graph.pt"))
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
	parser.add_argument('--sampling_rate', type=int, required=True,
					  help='Sampling rate of the data.')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	Path(args.output_directory).mkdir(parents=True, exist_ok=True)
	main(args)