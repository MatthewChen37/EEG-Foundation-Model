import mne, torch
import os, argparse
import numpy as np
import pandas as pd
import warnings
import traceback
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix
import pywt

def main(args):
    subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
    print("Subjects: ", len(subjects))
    errors = []

    with ThreadPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
        for future in tqdm(futures):
            result = future.result()
            if result is not None:
                errors.append(result)

    if len(errors) > 0:
        errors_df = pd.DataFrame(errors, columns=["subject", "error", "traceback"])
        errors_df.to_csv(os.path.join(args.input_directory, "graph_creation_errors.csv"), index=False)

def _process_subject(args, subject):
    base_path = os.path.join(args.input_directory, subject, "epochs_v2")

    if os.path.exists(base_path):
        try:
            # Create the output directory
            Path(os.path.join(args.input_directory, subject, "wavelet_decompositions")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(args.input_directory, subject, "graphs_v2")).mkdir(parents=True, exist_ok=True)

            
            for epoch_file in os.listdir(base_path):
                raw_epoch = mne.read_epochs(os.path.join(base_path, epoch_file), preload=True, verbose=False)
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
                    torch.save(data, os.path.join(args.input_directory, subject, "wavelet_decompositions", f"{epoch_file[:-4]}_{band}_band.pt"))

                # TODO: Add support for multiple features
                dist_feat = createDistanceMatrix(raw_epoch.info)
                electrode_pos = createPositionMatrix(raw_epoch.info)

                # Fully connected graph
                edge_indices, edge_weights = createEdges([dist_feat])

                # Create graph
                data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
                torch.save(data, os.path.join(args.input_directory, subject, "graphs_v2", f"{epoch_file[:-4]}_graph.pt"))
                return None
        except Exception as e:
            #print(f"Error processing {subject}: {e}, traceback: {traceback.format_exc()}")
            return (subject, e, traceback.format_exc())
    else:
        #print(f"Skipping {subject} as no epochs_v2 directory found")
        return None

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