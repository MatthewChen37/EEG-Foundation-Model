import sys
sys.path.append("../")
import mne, torch
import os, argparse
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor
from GenerateConnectivityGraphs import createDistanceMatrix, createEdges, createPositionMatrix
import pywt

def main(args):
    epoch_files = [f for f in os.listdir(args.input_directory) if f.endswith("_epo.fif")]
    print(f"Found {len(epoch_files)} epoched files in {args.input_directory}")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_epoch_file, args, epo_file) for epo_file in epoch_files]
        for future in tqdm(futures):
            future.result()

def process_epoch_file(args, epo_file):
    try:
        epo_path = os.path.join(args.input_directory, epo_file)
        labels_path = epo_path.replace("_epo.fif", "_labels.npy")

        if not os.path.exists(labels_path):
            print(f"Labels missing for {epo_file}, skipping")
            return

        # Load epochs and labels
        raw_epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        labels = np.load(labels_path)  # shape: (6, total_timepoints) or (epochs, 6, timepoints_per_epoch)
        
        # Pick channels
        eeg_raw = raw_epochs.copy().pick("eeg")
        gait_raw = raw_epochs.copy().pick("misc")

        n_epochs = len(raw_epochs)
        sfreq = raw_epochs.info['sfreq']

        # Make sure labels have the right shape: (n_epochs, n_channels, n_times)
        # If labels shape is (6, total_timepoints), segment into epochs:
        if labels.ndim == 2:
            # total_timepoints = labels.shape[1]
            samples_per_epoch = int(sfreq * 6)  # 6 seconds * sampling freq
            expected_points = samples_per_epoch * n_epochs
            if labels.shape[1] != expected_points:
                print(f"Label timepoints {labels.shape[1]} do not match epochs * 6 sec samples {expected_points}")
                return
            labels = labels.reshape((6, n_epochs, samples_per_epoch)).transpose(1,0,2)  # (epochs, channels, time)
        elif labels.shape[0] == n_epochs:
            # Already (epochs, channels, time)
            pass
        else:
            print(f"Unexpected label shape {labels.shape} for {epo_file}")
            return

        # Create output dirs
        base_name = epo_file[:-8]  # strip '-epo.fif'
        out_dir = os.path.join(args.output_directory, base_name)
        wavelet_dir = os.path.join(out_dir, "wavelet_decompositions")
        graph_dir = os.path.join(out_dir, "graphs")
        gait_dir = os.path.join(out_dir, "gait")
        Path(wavelet_dir).mkdir(parents=True, exist_ok=True)
        Path(graph_dir).mkdir(parents=True, exist_ok=True)
        Path(gait_dir).mkdir(parents=True, exist_ok=True)

        dist_feat = createDistanceMatrix(raw_epochs.info)
        electrode_pos = createPositionMatrix(raw_epochs.info)
        edge_indices, edge_weights = createEdges([dist_feat])

        for i in range(n_epochs):
            epoch_data = eeg_raw.get_data()[i]  # shape: (channels, time)
            epoch_gait = gait_raw.get_data()[i]  # misc channels for gait

            # Wavelet decomposition on EEG epoch
            wp = pywt.WaveletPacket(data=epoch_data, wavelet='db4', maxlevel=5, axis=-1)
            relevant_bands = {
                'delta': wp['aaaaa'].data,
                'theta': wp['aaaad'].data,
                'alpha': wp['aaad'].data,
                'beta': wp['aad'].data,
                'gamma': wp['ad'].data,
                'high_freq': wp['d'].data
            }

            # Save wavelet decompositions per band
            for band_name, band_data in relevant_bands.items():
                band_tensor = torch.tensor(band_data)
                torch.save(band_tensor, os.path.join(wavelet_dir, f"{base_name}_{band_name}_epoch_{i}.pt"))

            # Save gait data
            gait_tensor = torch.tensor(epoch_gait)
            torch.save(gait_tensor, os.path.join(gait_dir, f"{base_name}_gait_epoch_{i}.pt"))

            # Create graph data object
            # x can be anything, here just zeros of correct shape
            x = torch.zeros(epoch_data.shape[0], 1)  # (channels, features)
            data = Data(
                x=x,
                edge_index=edge_indices,
                edge_attr=edge_weights,
                pos=electrode_pos
            )

            # Attach the label time series for this epoch (shape: channels x time)
            label_tensor = torch.tensor(labels[i])
            data.y = label_tensor  # shape (channels, time)

            torch.save(data, os.path.join(graph_dir, f"{base_name}_graph_epoch_{i}.pt"))

    except Exception as e:
        print(f"Error processing {epo_file}: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Create graphs with aligned labels from MOBI epochs')
    parser.add_argument('--input_directory', type=str, required=True,
                        help='Directory containing MOBI epoch files (*.fif) and label files (*.npy)')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Output directory for generated graphs and features')
    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_args()
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    main(args)
