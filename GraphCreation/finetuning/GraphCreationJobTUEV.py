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


def intClass(class_name):
	if class_name == "spsw":
		return 0
	elif class_name == "gped":
		return 1
	elif class_name == "pled":
		return 2
	elif class_name == "eyem":
		return 3
	elif class_name == "artf":
		return 4
	elif class_name == "bckg":
		return 5
	else:
		raise ValueError(f"Unknown class {class_name}")

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def main(args):
	files = os.listdir(args.input_directory)
	files.remove("failed_files.csv")
	#assert len(files) == 154
	
	# Define the columns for the empty DataFrame
	columns = ['f_path', 'error', 'traceback'] 
	files_grouped = group_list(files, 30)
	print(f"Processing {len(files_grouped)} groups")

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_file_group, args, files_group) for files_group in files_grouped]
		for future in tqdm(futures):
			future.result()

def _process_file_group(args, files):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_file_thread, args, file) for file in files]
		for future in futures:
			future.result()

def _process_file_thread(args, file):
	base_path = os.path.join(args.input_directory, file)
	if os.path.exists(base_path):
		try:
			file_class = file.split("_")[0]
			epoch_idx = int(file[:-4].split("_")[-1])
			output_dir = os.path.join(args.output_directory, file[:-4])
			Path(os.path.join(output_dir, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(output_dir, "graphs")).mkdir(parents=True, exist_ok=True)
			raw = mne.io.read_raw_fif(base_path, preload=True, verbose=False)
			#annotations_per_epoch = raw_epoch.get_annotations_per_epoch()
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
			dist_feat = createDistanceMatrix(raw.info)
			electrode_pos = createPositionMatrix(raw.info)
			# Fully connected graph
			edge_indices, edge_weights = createEdges([dist_feat])
			#for epoch in range(len(annotations_per_epoch)):
				# if len(annotations_per_epoch[epoch]) > 0: # Only save epochs with annotations
			for band, data in relevant_bands.items():
				#assert data.shape[0] == len(annotations_per_epoch), f"Number of epochs {data.shape[0]} does not match number of annotations {len(annotations_per_epoch)}"
				# Save the wavelet decomposition
				curr_epoch = torch.tensor(data)
				torch.save(curr_epoch, os.path.join(output_dir, "wavelet_decompositions", f"{file[:-4]}_{band}_epoch_{epoch_idx}.pt"))

			# Create graph
			if args.split == "train":
				subject_int_class = int(file_class) - 1 # To match with the intClass function
			else:
				subject_int_class = intClass(file_class)
			data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
			data.y = torch.tensor([subject_int_class])
			torch.save(data, os.path.join(output_dir, "graphs", f"{file[:-4]}_graph_epoch_{epoch_idx}.pt"))
		except Exception as e:
			print(f"Failed to process {file}, error: {e}")

def process_annotation(annotations_for_epoch):
	'''
	Process the annotations for an epoch
	'''
	annotations = []
	for annotation in annotations_for_epoch:
		annotations.append([int(x) for x in annotation])
	return torch.tensor(annotations)

def parse_args():
	parser = argparse.ArgumentParser(description='Create graphs for TUAB')
	parser.add_argument('--input_directory', type=str, required=True,
					  help='Path to epoched data.')
	parser.add_argument('--output_directory', type=str, required=True,
					  help='Output directory for the graphs.')
	parser.add_argument('--split', type=str, default='train')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	Path(args.output_directory).mkdir(parents=True, exist_ok=True)
	main(args)