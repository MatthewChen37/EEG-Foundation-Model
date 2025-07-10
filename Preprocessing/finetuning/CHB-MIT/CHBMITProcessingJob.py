import mne
import sys
sys.path.append("../")
sys.path.append("../../")
import pandas as pd
import os, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from preprocessingPipeline import simplePipeline, group_list, simplePipelineNoEpoch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
from collections import defaultdict
import traceback
from convertTUHtoBIDS import _rename_channels, CHANNELS_TO_KEEP
import pickle as pkl
# Based on: https://github.com/ycq091044/BIOT/blob/main/datasets/CHB-MIT/process2.py

train_patients = {
    "chb01", "chb02", "chb03", "chb04",
	"chb05", "chb06", "chb07", "chb08",
    "chb09", "chb10", "chb11", "chb12",
    "chb13", "chb14", "chb15", "chb16",
    "chb17", "chb18", "chb19", "chb20",
}

val_patients = {
	"chb21", "chb22"
}

test_patients = {
	"chb23", "chb24"
}


def main(args):
	subjects = [f.path for f in os.scandir(args.input_dir) if f.is_dir()]
	print(f"Found {len(subjects)} subjects")

	# Define the columns for the empty DataFrame
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
		for future in tqdm(futures):
			future.result()

def _process_subject(args, subject):
	for file in os.listdir(subject):
		file_path = os.path.join(subject, file)
		recording = pkl.load(file_path)
		print("file_path:", type(recording))


		




def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess CHB-MIT data')
	parser.add_argument('--input_dir', type=str, help='Path to the directory containing the CHB-MIT data', default="/home/azureuser/mycontainer/CHB-MIT_Prepreprocessed")
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored', default="/home/azureuser/mycontainer/CHB-MIT_Preprocessed")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")
	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)