import mne
import gc
import numpy as np
import pandas as pd
import os, argparse
from pathlib import Path
from readEEG import HBN_ELECTRODE_MAP
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
from preprocessingPipeline import simplePipeline
from concurrent.futures import ProcessPoolExecutor
import warnings

ELECTRODES = [f'E{i}' for i in range(1, 129)]
TO_DROP = [electrode for electrode in ELECTRODES if electrode not in HBN_ELECTRODE_MAP.values()]
HBN_ELECTRODE_MAP_REVERSED = dict((v,k) for k,v in HBN_ELECTRODE_MAP.items())

def main(args):
	dataset_releases = os.listdir(args.input_directory)

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_dataset_release, dataset_release, args) for dataset_release in dataset_releases]
		for future in futures:
			future.result()

def _process_dataset_release(dataset_release, args):
	bids_path = BIDSPath(root=os.path.join(args.input_directory, dataset_release),
					    datatype="eeg", suffix="eeg", extension=".set")
		
	subjects = sorted(set([bp.subject for bp in bids_path.match()]))
	#print(f"Found {len(subjects)} subjects for dataset release: {dataset_release}")

	indices = None
	means = pd.DataFrame()
	stds = pd.DataFrame()

	for subject in subjects:
		#print(f"Processing subject: {subject} in dataset release: {dataset_release}")
		Path(os.path.join(args.output_dir, subject)).mkdir(parents=True, exist_ok=True)
		Path(os.path.join(args.output_dir, subject, "epochs")).mkdir(parents=True, exist_ok=True)
		subject_all_data = []
		bids_path.update(subject=subject)
		for bp in tqdm(bids_path.match()):
			if not os.path.exists(file_path):
				if bp.run:
					file_path = os.path.join(args.output_dir, subject, "epochs", f"{bp.task}-{bp.run}.fif")
				else:
					file_path = os.path.join(args.output_dir, subject, "epochs", f"{bp.task}.fif")

				raw = read_raw_bids(bp, extra_params={'preload':True}, verbose=False)
				raw.drop_channels(TO_DROP)
				raw.rename_channels(HBN_ELECTRODE_MAP_REVERSED)
				if indices is None:
					indices = raw.ch_names
				subject_all_data.append(raw.get_data())

				epochs = simplePipeline(raw)
				epochs.save(file_path, overwrite=False)

		subject_all_data = np.concatenate(subject_all_data, axis=1)

		# Z-score Transform By Channel
		mean = np.mean(subject_all_data, axis=1)
		std = np.std(subject_all_data, axis=1)

		means[subject] = mean
		stds[subject] = std

	means.set_index(pd.Index(indices), inplace=True)
	stds.set_index(pd.Index(indices), inplace=True)

	means.to_csv(os.path.join(args.output_dir, f"{dataset_release}_mean.csv"))
	stds.to_csv(os.path.join(args.output_dir, f"{dataset_release}_std.csv"))

	gc.collect()

def parse_args():
	# setup arg parser
	parser = argparse.ArgumentParser()

	# INPUT arguments 
	parser.add_argument(
		"--input_directory", type=str, help="Directory where all HBN datasets are stored", required=True
	)

	parser.add_argument(
		"--output_dir", type=str, help="Output directory", required=True
	)

	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main(args)