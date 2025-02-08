import mne
import sys
sys.path.append("../")
import pandas as pd
import os, argparse
from pathlib import Path
from tqdm import tqdm
from preprocessingPipeline import simplePipeline
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import traceback
from collections import OrderedDict
from convertTUHtoBIDS import _rename_channels, CHANNELS_TO_KEEP

INDICES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
		    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
			  'Fz', 'Cz', 'Pz']

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
		session_folders = os.listdir(subclass_folder)
		for session_folder in session_folders:
			files = os.listdir(os.path.join(subclass_folder, session_folder))
			Path(os.path.join(args.output_dir, c)).mkdir(parents=True, exist_ok=True)
			files_grouped = group_list(files, 8)
			with ProcessPoolExecutor() as executor:
				futures = [executor.submit(_process_file, args, c, session_folder, files_group) for files_group in files_grouped]
				for future in tqdm(futures):
					future.result()
	failed_files_df = pd.DataFrame(failed_files, columns=columns)
	failed_files_df.to_csv(os.path.join(args.output_directory, "failed_files.csv"), index=False)

def _process_file(args, c, session_folder, files):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_file_thread, args, c, session_folder, file) for file in files]
		for future in tqdm(futures):
			result = future.result()
			if result is not None:
				failed_files.append(result)
	return None

def _process_file_thread(args, c, session_folder, file_name):
	file_output_path = os.path.join(args.output_dir, c, f"{file_name}_epo.fif")
	if not os.path.exists(file_output_path):
		try:
			raw = mne.io.read_raw_edf(os.path.join(args.input_directory, c, session_folder, file_name), preload=True).copy()
			_rename_channels(raw)
			to_drop = [ch for ch in raw.ch_names if ch not in CHANNELS_TO_KEEP]
			raw.drop_channels(to_drop)
			raw.info["line_freq"] = 60
			raw.set_montage("standard_1005", on_missing="ignore")
			raw.set_meas_date(None)
			ch_names = raw.ch_names
			if 'A1' and 'A2' in ch_names:
				raw = raw.drop_channels(['A1', 'A2'])
			assert len(raw.ch_names) == 19, f"Number of channels is {len(raw.ch_names)}"

			epochs = simplePipeline(raw)
			epochs.save(file_output_path, overwrite=False)
			return None
		except Exception as e:
			print(f"Failed to process {file_name}, error: {e}")
			return (file_name, e, traceback.format_exc())
					
def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess TUH EEG data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the TUH EEG data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)

