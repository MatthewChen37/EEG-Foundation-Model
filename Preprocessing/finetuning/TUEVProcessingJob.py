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

def convertTUEV(subject_dir, subject_name):
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
					ch_names = [_channel_mapping(x.split("-")) for x in annotations_df['channel'].values.tolist()]
					
					global_annotations_df = pd.read_csv(global_annotation_file_path, delimiter=',', comment='#')
					onset = pd.concat([onset, global_annotations_df['start_time']])
					duration = pd.concat([duration, global_annotations_df['stop_time'] - global_annotations_df['start_time']])
					description = pd.concat([description, global_annotations_df['label']])
					global_channel_names = [[] for x in global_annotations_df['channel'].values.tolist()]
					ch_names = ch_names + global_channel_names
					
					annotations = mne.Annotations(onset=onset, duration=duration, description=description, ch_names=ch_names)


					raw.set_annotations(annotations)
					raw.set_montage("standard_1005", on_missing="ignore")

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