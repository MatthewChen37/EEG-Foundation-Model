# %%
import sys
from mne.io import concatenate_raws
sys.path.append("./")
sys.path.append("../../")
from edf_ import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from preprocessingPipeline import simplePipeline, group_list, simplePipelineNoEpoch
from mne import make_fixed_length_epochs
import traceback

# Based on: https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/ISRUC/prepare_ISRUC_1.py

def main(args):
	psg_f_names = []
	label_f_names = []
	for i in range(1, 101):
		numstr = str(i)
		psg_f_names.append(f'{args.input_dir}/{numstr}.rec')
		label_f_names.append(f'{args.input_dir}/{numstr}_1.txt')

	psg_label_f_pairs = []
	for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
		if psg_f_name[:-4] == label_f_name[:-6]:
			psg_label_f_pairs.append((psg_f_name, label_f_name))
	
	'''
	for item in psg_label_f_pairs:
		print(item)
	'''

	label2id = {'0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '5': 4}
	print(label2id)

	#print(f"Found {len(psg_label_f_pairs)} pairs")

	for subject_idx in tqdm(range(len(psg_label_f_pairs))):
		process_subject(args, psg_label_f_pairs[subject_idx][0], psg_label_f_pairs[subject_idx][1], subject_idx + 1, label2id)

def _rename_channels_isruc(raw):
	channels_to_keep = {
		'F3-A2',
		'C3-A2',
		'O1-A2',
		'F4-A1',
		'C4-A1',
		'O2-A1',
	}

	to_drop = [ch for ch in raw.ch_names if ch not in channels_to_keep]
	raw.drop_channels(to_drop)
	new_mapping = {
		'F3-A2': 'F3',
		'C3-A2': 'C3',
		'O1-A2': 'O1',
		'F4-A1': 'F4',
		'C4-A1': 'C4',
		'O2-A1': 'O2',
	}
	raw.rename_channels(new_mapping)

def process_subject(args, psg_f_name, label_f_name, subject_idx, label2id):
	print(f"Processing subject {subject_idx}")
	raw = read_raw_edf(psg_f_name, preload=True)

	_rename_channels_isruc(raw)
	raw.info["line_freq"] = 60
	raw.set_montage("standard_1005", on_missing="ignore", match_case=False)
	raw.set_meas_date(None)
	ch_names = raw.ch_names
	if 'A1' and 'A2' in ch_names:
		raw = raw.drop_channels(['A1', 'A2'])
	assert len(raw.ch_names) == 6, f"Number of channels is {len(raw.ch_names)}"

	raw = simplePipelineNoEpoch(raw, sample_rate=128, low_pass=75)
	epochs = make_fixed_length_epochs(raw, duration=30, preload=True)

	labels_list = []
	for line in open(label_f_name).readlines():
		line_str = line.strip()
		if line_str != '':
			labels_list.append(label2id[line_str])

	assert epochs.get_data().shape[0] == len(labels_list), f"Epochs shape {epochs.get_data().shape[0]} does not match labels length {len(labels_list)}"
	labels = np.array(labels_list)

	'''
	epoch_data = epochs.get_data()

	for i in range(len(labels)):
		epoch = epoch_data[i]
		label = labels[i]
		subject_output_directory = os.path.join(args.output_dir, "train" if subject_idx <= 80 else "val" if subject_idx <= 90 else "test", str(subject_idx))
		Path(subject_output_directory).mkdir(parents=True, exist_ok=True)
		file_output_path = os.path.join(subject_output_directory, f"{subject_idx}_{i}_{label}.fif")
		raw_epoch = mne.io.RawArray(epoch, raw.info)
		raw_epoch.save(file_output_path, overwrite=True)
		sys.exit(0)
	'''
	
def parse_args():
	parser = argparse.ArgumentParser(description="ISRUC Processing Job")
	parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the ISRUC dataset", default="/home/azureuser/mycontainer/ISRUC")
	parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
	return parser.parse_args()


if __name__ == "__main__":
    # Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)
