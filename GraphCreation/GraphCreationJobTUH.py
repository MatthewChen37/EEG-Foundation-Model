import mne, torch
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
import sys
sys.path.append("../Preprocessing")
from preprocessingPipeline import group_list

def main(args):
    subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
    print("Subjects: ", len(subjects))

    subject_groups = group_list(subjects, 30)

    errors = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_subject_group, args, subject_group) for subject_group in subject_groups]
        for future in tqdm(futures):
            result = future.result()
            if result is not None:
                errors.extend(result)

    if len(errors) > 0:
        print(f"Failed to process {len(errors)} subjects")
        errors_df = pd.DataFrame(errors, columns=["subject", "error", "traceback"])
        errors_df.to_csv(os.path.join(args.input_directory, f"graph_creation_errors_v{args.version}.csv"), index=False)

def _process_subject_group(args, subjects):
    group_errors = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
        for future in futures:
            result = future.result()
            if result is not None:
                group_errors.append(result)
    return group_errors

def _process_subject(args, subject):
    base_path = os.path.join(args.input_directory, subject, f"epochs_v{args.version}")

    if os.path.exists(base_path):
        try:
            # Create the output directory
            Path(os.path.join(args.input_directory, subject, f"wavelet_decompositions_v{args.version}")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(args.input_directory, subject, f"graphs_v{args.version}")).mkdir(parents=True, exist_ok=True)

            
            for epoch_file in os.listdir(base_path):
                raw_epoch = mne.read_epochs(os.path.join(base_path, epoch_file), preload=True, verbose=False)
                raw_data = raw_epoch.get_data(copy=True)

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
                    data = torch.tensor(data)
                    torch.save(data, os.path.join(args.input_directory, subject, f"wavelet_decompositions_v{args.verison}", f"{epoch_file[:-4]}_{band}_band.pt"))

                # TODO: Add support for multiple features
                dist_feat = createDistanceMatrix(raw_epoch.info)
                electrode_pos = createPositionMatrix(raw_epoch.info)

                # Fully connected graph
                edge_indices, edge_weights = createEdges([dist_feat])

                # Create graph
                data = Data(x=raw_data[0], edge_index=edge_indices, edge_attr=edge_weights, pos=electrode_pos)
                torch.save(data, os.path.join(args.input_directory, subject, f"graphs_v{args.version}", f"{epoch_file[:-4]}_graph.pt"))
                return None
        except Exception as e:
            return (subject, e, traceback.format_exc())
    else:
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize based on mean/std and create graphs')
    parser.add_argument('--input_directory', type=str, required=True,
					  help='Path to BIDS directory containing the preprocessed data. Processing will also write to this directory.')
    parser.add_argument('--version', type=str, help='Version of the preprocessing')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	main(args)