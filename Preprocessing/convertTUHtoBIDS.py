import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
import pandas as pd
import numpy as np
import os

def convertTUHtoBIDS(filepath):
	pass


def convertTUSZtoBIDS(subject_dir, subject_name):
	
	for session in os.listdir(subject_dir):
		for montage_layout in os.listdir(os.path.join(subject_dir, session)):
			for file in os.listdir(os.path.join(subject_dir, session, montage_layout)):
				if file.endswith(".edf"):
					raw_filepath = os.path.join(subject_dir, session, montage_layout, file)
					channel_annotation_file_path = os.path.join(subject_dir, session, montage_layout, file[:-4] + ".csv")
					global_annotation_file_path = os.path.join(subject_dir, session, montage_layout, file[:-4] + ".csv_bi")

					raw = mne.io.read_raw_edf(raw_filepath, preload=False)
					_rename_channels(raw)

					'''
					BIDS Format requires line frequency to be specified.
					Line frequency is the frequency of the power line in the country where the data was recorded.
					For the United States, the line frequency is (typically) 60 Hz.	
					'''
					raw.info["line_freq"] = 60

					# Handling annotations
					annotations_df = pd.read_csv(channel_annotation_file_path, delimiter=',', comment='#')
					onset = annotations_df['start_time']
					duration = annotations_df['stop_time'] - annotations_df['start_time']
					description = annotations_df['label']
					ch_names = [x.split("-") for x in annotations_df['channel'].values.tolist()]
					
					global_annotations_df = pd.read_csv(global_annotation_file_path, delimiter=',', comment='#')
					onset = pd.concat([onset, global_annotations_df['start_time']])
					duration = pd.concat([duration, global_annotations_df['stop_time'] - global_annotations_df['start_time']])
					description = pd.concat([description, global_annotations_df['label']])
					global_channel_names = [[] for x in global_annotations_df['channel'].values.tolist()]
					ch_names = ch_names + global_channel_names
					
					annotations = mne.Annotations(onset=onset, duration=duration, description=description, ch_names=ch_names)


					raw.set_annotations(annotations)
					raw.set_montage("standard_1005", on_missing="ignore")

					bids_path = BIDSPath(subject="aaaaaaac", session=session, run=file[:-4].split("_")[-1], task="Rest", root="SampleData/TUH-BIDS")
					write_raw_bids(raw, bids_path, overwrite=True)

def _channel_mapping(channel_list):
	new_list = []

	for channel in channel_list:
		if channel == "FP1":
			new_list.append("Fp1")
		elif channel == "FP2":
			new_list.append("Fp2")
		elif channel == "CZ":
			new_list.append("Cz")
		else:
			new_list.append(channel)
	return new_list


# From https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/tuh.py
def _rename_channels(raw):
	"""
	Renames the EEG channels using mne conventions and sets their type to 'eeg'.

	See https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
	"""
	# remove ref suffix and prefix:
	# TODO: replace with removesuffix and removeprefix when 3.8 is dropped
	mapping_strip = {
		c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
		for c in raw.ch_names
	}
	raw.rename_channels(mapping_strip)

	montage1005 = mne.channels.make_standard_montage("standard_1005")

	# Changed this line
	mapping_eeg_names = {
		c.upper(): c for c in montage1005.ch_names if c.upper() in raw.ch_names
	}

	# Set channels whose type could not be inferred (defaulted to "eeg") to "misc":
	non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
	if non_eeg_names:
		non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
		mapping_non_eeg_types = {
			c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
		}
		if mapping_non_eeg_types:
			raw.set_channel_types(mapping_non_eeg_types)

	if mapping_eeg_names:
		# Set 1005 channels type to "eeg":
		raw.set_channel_types(
			{c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore"
		)
		# Fix capitalized EEG channel names:
		raw.rename_channels(mapping_eeg_names)