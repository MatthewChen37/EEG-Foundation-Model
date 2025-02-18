import pandas as pd
import os, argparse, mne
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

import sys
sys.path.append("../")
from preprocessingPipeline import simplePipeline

CHANNELS_TO_KEEP = {
	"Fp1",
	"Fp2",
	"F3",
	"F4",
	"C3",
	"C4",
	"A1",
	"A2",
	"P3",
	"P4",
	"O1",
	"O2",
	"F7",
	"F8",
	"T3",
	"T4",
	"T5",
	"T6",
	"Fz",
	"Cz",
	"Pz",
}

session_start_times = {
	1: [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
	2: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
	3: [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
}

session_end_times = {
	1: [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
	2: [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
	3: [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066],
}

def main(args):
	files = os.listdir(args.input_directory)
	files = [file for file in files if file.endswith(".cnt")]
	assert len(files) == 48, f"Expected 49 files, found {len(files)}"

	scores = pd.read_csv('/home/azureuser/cloudfiles/code/Users/mchen439/EEG-Foundation-Model/Preprocessing/finetuning/Scores.csv')
	scores = scores.iloc[0:48]
	scores.columns = ["Subject", "Experiment Index", "Trial 0", "Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Trial 6", "Trial 7", "Trial 8", "Trial 9", "Trial 10", "Trial 11", "Trial 12", "Trial 13", "Trial 14"]

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_file, args, file, scores) for file in files]
		for future in tqdm(futures):
			future.result()

def _process_file(args, file, scores):
	try:
		subject = int(file.split("_")[0])
		session_number = int(file.split("_")[1])

		# Get the start and end times for the session
		trial_start_times = session_start_times[session_number]
		trial_end_times = session_end_times[session_number]

		# Load the raw data
		raw = mne.io.read_raw_cnt(os.path.join(args.input_directory, file), preload=True, verbose=False)

		channels_to_drop = [ch_name for ch_name in raw.ch_names if ch_name not in CHANNELS_TO_KEEP]
		channels_to_drop.remove('FP1')
		channels_to_drop.remove('FP2')
		channels_to_drop.remove('FZ')
		channels_to_drop.remove('CZ')
		channels_to_drop.remove('PZ')
		channels_to_drop.remove('C5')
		channels_to_drop.remove('C6')
		channels_to_drop.remove('P5')
		channels_to_drop.remove('P6')
		channels_to_drop += ['M1', 'M2', 'VEO', 'HEO']

		raw.drop_channels(channels_to_drop)


		to_lowercase = {"FP1": "Fp1",
			"FP2": "Fp2",
			"FZ": "Fz",
			"CZ": "Cz",
			"PZ": "Pz"
		}

		raw.rename_channels(to_lowercase)

		custom_montage = mne.channels.read_custom_montage('/home/azureuser/mycontainer/SEED-V/SEED-V/channel_62_pos.locs')
		raw.set_montage(custom_montage)

		mapping = {
			"C5": "T3",
			"C6": "T4",
			"P5": "T5",
			"P6": "T6",
		}
			
		raw.rename_channels(mapping)

		Path(os.path.join(args.output_dir, str(subject))).mkdir(parents=True, exist_ok=True)
		for trial_number, (start_time, end_time) in enumerate(zip(trial_start_times, trial_end_times)):
			label = mne.Annotations(onset=0, duration=end_time-start_time, description=scores[scores["Subject"] == subject]["Trial " + str(trial_number)].iloc[session_number-1])
			raw_copy = raw.copy().crop(tmin=start_time, tmax=end_time)
			raw_copy.set_annotations(label)
			try:
				epochs = simplePipeline(raw_copy)
				epochs.save(os.path.join(args.output_dir, f"sub-{subject}_ses-{session_number}_run-{trial_number}_eeg.fif"), overwrite=True)
			except ValueError as e:
				print(f"Skipping session {session_number} trial {trial_number} because of error: {e}")
				continue
	except Exception as e:
		print(f"Failed to process {file}")
		raise e


def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess seedV EEG data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the seedV EEG data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)

