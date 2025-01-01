import mne, torch
import os, argparse
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix

def main(args):
	agg_mean = pd.read_csv(os.path.join(args.input_directory, 'TUH_means.csv'), index_col=0) 
	agg_std = pd.read_csv(os.path.join(args.input_directory, 'TUH_stds.csv'), index_col=0)

	subjects = [f.path for f in os.scandir(args.input_directory) if f.is_dir()]

	'''
	Although Pandas Dataframes are not thread-safe,
	we are only reading from them in this function, so it is safe to use them in a ThreadPoolExecutor.
	'''
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject, agg_mean, agg_std) for subject in subjects]
		for future in tqdm(futures):
			future.result()

def _process_subject(args, subject, agg_mean, agg_std):
    # Create the output directory
    Path(os.path.join(args.input_directory, subject, "normalized_epochs")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.input_directory, subject, "graphs")).mkdir(parents=True, exist_ok=True)

    base_path = os.path.join(args.input_directory, subject, "epochs")

    for epoch_file in os.listdir(base_path):
        raw_epoch = mne.read_epochs(os.path.join(base_path, epoch_file), preload=True, verbose=False)
        annotations = raw_epoch.get_annotations_per_epoch()

        raw_data = _normalize_data(raw_epoch.get_data(copy=True), subject, agg_mean, agg_std)

        # Replace all NaNs with 0
        raw_data = np.nan_to_num(raw_epoch)

        np.save(os.path.join(args.input_directory, subject, "normalized_epochs", epoch_file), raw_data)

        # TODO: Add support for multiple features
        dist_feat = createDistanceMatrix(raw_epoch.info)
        electrode_pos = createPositionMatrix(raw_epoch.info)

        # Fully connected graph
        edge_indices, edge_weights = createEdges([dist_feat])

        # Create graphs
        graphs = []

        for i in range(raw_data.shape[0]):
            data = Data(x=raw_data[i], edge_index=edge_indices, edge_attr=edge_weights, y=annotations[i], pos=electrode_pos)
            graphs.append(data)

        # Each epoch is saved as a separate graph
        for idx, graph in enumerate(graphs):
            torch.save(graph, os.path.join(args.input_directory, subject, "graphs", f"{epoch_file[:-4]}_epoch_{idx}.pt"))

    return

def _normalize_data(data, subject, mean, std):
	return (data - mean[subject].to_numpy()[:, None]) / std[subject].to_numpy()[:, None]


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