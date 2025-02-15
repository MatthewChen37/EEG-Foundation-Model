import sys
sys.path.append("../")
import mne, torch
import os, argparse
import numpy as np
import pandas as pd
import warnings
import traceback
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.data import Data
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix
import pywt

def main(args):
    subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
    print("Subjects: ", len(subjects))

    with ProcessPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
        for future in tqdm(futures):
            future.result()

def _process_subject(args, subject):
	base_path = os.path.join(args.input_directory, subject, "epochs")

	if os.path.exists(base_path):
		try:
            # Create the output directory
			Path(os.path.join(args.input_directory, subject, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(args.input_directory, subject, "graphs")).mkdir(parents=True, exist_ok=True)
			Path(os.path.join(args.input_directory, subject, "gait")).mkdir(parents=True, exist_ok=True)

			for epoch_file in os.listdir(base_path):
				raw_epoch = mne.read_epochs(os.path.join(base_path, epoch_file), preload=True, verbose=False)
				eeg_raw = raw_epoch.copy().pick("eeg")
				gait_raw = raw_epoch.copy().pick("misc")

				raw_data = eeg_raw.get_data(copy=True)
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
					torch.save(torch.tensor(data), os.path.join(args.input_directory, subject, "wavelet_decompositions", f"{epoch_file[:-4]}_{band}_band.pt"))
				
				torch.save(torch.tensor(gait_raw.get_data(copy=True)), os.path.join(args.input_directory, subject, "gait", f"{epoch_file[:-4]}_gait.pt"))

				# TODO: Add support for multiple features
				dist_feat = createDistanceMatrix(raw_epoch.info)
				electrode_pos = createPositionMatrix(raw_epoch.info)

				# Fully connected graph
				edge_indices, edge_weights = createEdges([dist_feat])

				# Create graph
				data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
				torch.save(data, os.path.join(args.input_directory, subject, "graphs", f"{epoch_file[:-4]}_graph.pt"))
		except Exception as e:
			print(f"Error processing {subject}: {e}, traceback: {traceback.format_exc()}")
	

def parse_args():
	parser = argparse.ArgumentParser(description='Normalize based on mean/std and create graphs')
	parser.add_argument('--input_directory', type=str, required=True,
					  help='Path to BIDS directory containing the preprocessed data. Processing will also write to this directory.')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	main(args)