import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessingPipeline import simplePipeline
import os, argparse
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def main(args):
	subfolders = [f.path for f in os.scandir(args.input_directory) if f.is_dir()]
	subjects = [x.split("/")[-1][4:] for x in subfolders]
	print(f"Found {len(subjects)} subjects")

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
		for future in tqdm(futures):
			future.result()

def _process_subject(args, subject):
	Path(os.path.join(args.output_dir, subject)).mkdir(parents=True, exist_ok=True)
	Path(os.path.join(args.output_dir, subject, "epochs")).mkdir(parents=True, exist_ok=True)

	sessions = [f.path for f in os.scandir(os.path.join(args.input_directory, "sub-" + subject)) if f.is_dir()]
	sessions = [x.split("/")[-1][4:] for x in sessions]
	for session in sessions:
		bids_path = BIDSPath(root=args.input_directory, subject=subject, session=session, extension=".eeg")
		for bp in bids_path.match():
			if bp.fpath.suffix == ".vhdr":
				raw = read_raw_bids(bp, verbose=False, extra_params={'preload':True})
				assert len(raw.ch_names) == 19, f"Number of channels is {len(raw.ch_names)}"
				joints = pd.read_csv(os.path.join(bp.directory, f"{subject}_{session}_joints.csv"))

				joints_np = joints.to_numpy().T
				info = mne.create_info(ch_names=["GHR", "GKR", "GAR", "GHL", "GKL", "GAL"], sfreq=raw.info["sfreq"], ch_types=["misc" for i in range(6)])
				joints_raw = mne.io.RawArray(joints_np, info, verbose=False)

				raw.add_channels([joints_raw], force_update_info=True)

				epochs = simplePipeline(raw)
				#print(raw.get_channel_types())
				epochs.save(os.path.join(args.output_dir, subject, "epochs", f"{session}_epo.fif"), overwrite=)

def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess MOBI data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the MOBI data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)