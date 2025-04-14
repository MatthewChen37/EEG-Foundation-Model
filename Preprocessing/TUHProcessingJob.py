import mne
import gc
import numpy as np
import pandas as pd
import os, argparse
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids, get_bids_path_from_fname
from tqdm import tqdm
from preprocessingPipeline import simplePipeline, group_list
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import traceback
import pickle as pkl
from collections import OrderedDict

INDICES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
		    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
			  'Fz', 'Cz', 'Pz']

def main(args):
	subfolders = [f.path for f in os.scandir(args.input_directory) if f.is_dir()]
	subjects = [x.split("/")[-1][4:] for x in subfolders]
	print(f"Found {len(subjects)} subjects")

	# Define the columns for the empty DataFrame
	columns = ['bids_path', 'error', 'traceback'] 

	subjects_grouped = group_list(subjects, 30)
	print(f"Grouped subjects into {len(subjects_grouped)} groups")


	failed_files = []
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject_group, args, subject_group) for subject_group in subjects_grouped]
		for future in tqdm(futures):
			result = future.result()
			failed_files.extend(result)

	if len(failed_files) > 0:
		print(f"Failed to process {len(failed_files)} files")
		failed_files_df = pd.DataFrame(failed_files, columns=columns)
		failed_files_df.to_csv(os.path.join(args.output_dir, f"failed_files_preprocessing_v{args.version}.csv"), index=False)

def _process_subject_group(args, subject_group):
	failed_subjects = []
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subject_group]
		for future in futures:
			result = future.result()
			if result is not None:
				failed_subjects.append(result)
	return failed_subjects

def _process_subject(args, subject):
	Path(os.path.join(args.output_dir, subject)).mkdir(parents=True, exist_ok=True)
	Path(os.path.join(args.output_dir, subject, f"epochs_v{args.version}")).mkdir(parents=True, exist_ok=True)

	bids_path = BIDSPath(root=os.path.join(args.input_directory + f"/sub-{subject}"),
					   datatype="eeg", suffix="eeg", extension=".edf")
	
	for bp in bids_path.match():
		bp = _correct_path(bp)
		file_path = os.path.join(args.output_dir, subject, f"epochs_v{args.version}", f"{bp.session}-{bp.processing}-{bp.recording}_epo.fif")
		if not os.path.exists(file_path):
			try:
				raw = read_raw_bids(bp, extra_params={'preload':True}, verbose=False).copy()
				ch_names = raw.ch_names
				if 'A1' and 'A2' in ch_names:
					raw = raw.drop_channels(['A1', 'A2'])
				assert len(raw.ch_names) == 19, f"Number of channels is {len(raw.ch_names)}"
				epochs = simplePipeline(raw, sample_rate=256, low_pass=120)
				epochs.save(file_path, overwrite=True)
				return None
			except Exception as e:
				print(f"Failed to process {bp}, error: {e}")
				return (bp, e, traceback.format_exc())
			

					
def _correct_path(bp):
	path_str = str(bp.fpath)

	parts = path_str.split("/")
	parts = list(OrderedDict.fromkeys(parts))

	new_path = Path("/".join(parts))

	return get_bids_path_from_fname(new_path)

def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess TUH EEG data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the TUH EEG data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	parser.add_argument('--version', type=str, help='Version of the preprocessing')

	args = parser.parse_args()
	return args

if __name__ == '__main__':

	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	main(args)