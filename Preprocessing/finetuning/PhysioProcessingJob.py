import mne
import sys
sys.path.append("../")
import pandas as pd
import os, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from preprocessingPipeline import simplePipeline, group_list, simplePipelineEventsFromAnnotations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import traceback
from convertTUHtoBIDS import CHANNELS_TO_KEEP
# Adapted from https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_physio.py

tasks = ['04', # left and right fist
		 '06', # both fists or both feet
		 '08', # left and right fist
		 '10', # both fists or both feet
		 '12', # left and right fist
		 '14'  # both fists or both feet
] # select the data for motor imagery


def main(args):
	subjects = os.listdir(args.input_dir)
	print(f"Found {len(subjects)} subjects")

	for subject in tqdm(subjects):
		subject_path = os.path.join(args.input_dir, subject)
		if os.path.isdir(subject_path):
			process_subject(args, subject)
		else:
			print(f"Skipping {subject}, not a directory")

def _rename_channels_eegmmidb(raw):
	new_mapping = dict()
	for ch in raw.ch_names:
		new_mapping[ch] = ch.replace(".", "")
		new_mapping[ch] = new_mapping[ch].upper()

		if new_mapping[ch] == "T7":
			new_mapping[ch] = "T3"
		elif new_mapping[ch] == "T8":
			new_mapping[ch] = "T4"
		elif new_mapping[ch] == "T9":
			new_mapping[ch] = "A1"
		elif new_mapping[ch] == "T10":
			new_mapping[ch] = "A2"
		elif new_mapping[ch] == "P5":
			new_mapping[ch] = "T5"
		elif new_mapping[ch] == "P6":
			new_mapping[ch] = "T6"
	raw.rename_channels(new_mapping)


def process_subject(args, subject):
	subject_idx = int(subject[1:])
	if subject_idx < 70: # Train
		subject_output_directory = os.path.join(args.output_dir, "train")
	elif subject_idx >= 70 and subject_idx < 89: # Validation
		subject_output_directory = os.path.join(args.output_dir, "val")
	else: # Test
		subject_output_directory = os.path.join(args.output_dir, "test")

	Path(subject_output_directory).mkdir(parents=True, exist_ok=True)
	subject_output_path = os.path.join(subject_output_directory, subject)
	Path(subject_output_path).mkdir(parents=True, exist_ok=True)

	subject_path = os.path.join(args.input_dir, subject)
	for task in tasks:
		file = f"{subject}R{task}.edf"
		file_path = os.path.join(subject_path, file)
		try:
			raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False).copy()
			_rename_channels_eegmmidb(raw)
			channels_to_keep_upper = {ch.upper() for ch in CHANNELS_TO_KEEP}
			to_drop = [ch for ch in raw.ch_names if ch not in channels_to_keep_upper]
			raw.drop_channels(to_drop)
			raw.info["line_freq"] = 60
			raw.set_montage("standard_1005", on_missing="ignore", match_case=False)
			raw.set_meas_date(None)
			ch_names = raw.ch_names
			if 'A1' and 'A2' in ch_names:
				raw = raw.drop_channels(['A1', 'A2'])
			assert len(raw.ch_names) == 19, f"Number of channels is {len(raw.ch_names)}"

			events_from_annot, event_dict = mne.events_from_annotations(raw, verbose=False)
			epochs = simplePipelineEventsFromAnnotations(raw, events_from_annot, event_dict, sample_rate=128, low_pass=64)
			data = epochs.get_data()
			events = epochs.events[:, 2]
			data = data[:, :, -20 * 128:]

			for i, (sample, event) in enumerate(zip(data, events)):
				if event != 1: # 1 corresponds to rest
					label = event  - 2 if task in ['04', '06', '12'] else event # TODO: What does this line mean? I think its a smart way of getting the label
					raw_annotation_event = mne.io.RawArray(sample, info=raw.info, verbose=False)
					raw_annotation_event.set_annotations(mne.Annotations(onset=10, duration=4, description=str(label)))
					raw_annotation_event.save(os.path.join(subject_output_path, f"{file[:-4]}_event_{i}.fif"), overwrite=True)

		except Exception as e:
			traceback.print_exc()
			print(f"Failed to process {file}: {e}")


def parse_args():
	parser = argparse.ArgumentParser(description="Process PhysioNet data")
	parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the PhysioNet data")
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
	return parser.parse_args()

if __name__ == "__main__":
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)