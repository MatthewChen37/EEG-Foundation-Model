import mne
import gc
import numpy as np
import pandas as pd
import os, argparse
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
from preprocessingPipeline import simplePipeline, group_list
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import traceback
import pickle as pkl

ELECTRODES = [f'E{i}' for i in range(1, 129)]
HBN_ELECTRODE_MAP = {
	'Fp1': 'E22',
	'Fp2': 'E9',
	'F7': 'E33',
	'F3': 'E24',
	'Fz': 'E11',
	'F4': 'E124',
	'F8': 'E122',
	'T3': 'E45',
	'C3': 'E36',
	'Cz': 'Cz',
	'C4': 'E104',
	'T4': 'E108',
	'T5': 'E58',
	'P3': 'E52',
	'Pz': 'E62',
	'P4': 'E92',
	'T6': 'E96',
	'O1': 'E70',
	'O2': 'E83',
}
TO_DROP = [electrode for electrode in ELECTRODES if electrode not in HBN_ELECTRODE_MAP.values()]
HBN_ELECTRODE_MAP_REVERSED = dict((v,k) for k,v in HBN_ELECTRODE_MAP.items())

failed_files = []
def main(args):
	dataset_releases = os.listdir(args.input_directory)
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_dataset_release, dataset_release, args) for dataset_release in dataset_releases]
		for future in tqdm(futures):
			result = future.result()
			failed_files.extend(result)

	if failed_files:
		print(f"Failed to process {len(failed_files)} files")
		failed_files_df = pd.DataFrame(failed_files, columns=["subject", "error", "file_path", "traceback"])
		_write_csv(failed_files_df, os.path.join(args.output_dir, f"failed_files_v{args.version}.csv"))

def _process_dataset_release(dataset_release, args):
	bids_path = BIDSPath(root=os.path.join(args.input_directory, dataset_release),
					    datatype="eeg", suffix="eeg", extension=".set")
	subjects = sorted(set([bp.subject for bp in bids_path.match()]))
	failed_release_files = []
	# Group subjects into groups of 30
	subjects_grouped = group_list(subjects, 30)

	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_dataset_release_thread, subject_group, dataset_release, bids_path.copy(), args, failed_release_files) for subject_group in subjects_grouped]
		for future in futures:
			future.result()
	return failed_release_files

def _process_dataset_release_thread(subject_group, dataset_release, bids_path, args, failed_release_files):
	indices = None
	for subject in subject_group:
		Path(os.path.join(args.output_dir, subject)).mkdir(parents=True, exist_ok=True)
		Path(os.path.join(args.output_dir, subject, f"epochs_v{args.version}")).mkdir(parents=True, exist_ok=True)
		bids_path.update(subject=subject)
		for bp in bids_path.match():
			if bp.run:
				file_path = os.path.join(args.output_dir, subject, f"epochs_v{args.version}", f"{bp.task}-{bp.run}.fif")
			else:
				file_path = os.path.join(args.output_dir, subject, f"epochs_v{args.version}", f"{bp.task}.fif")
			if not os.path.exists(file_path):
				try:		
					raw = read_raw_bids(bp, extra_params={'preload':True}, verbose=False)
					raw.drop_channels(TO_DROP)
					raw.rename_channels(HBN_ELECTRODE_MAP_REVERSED)
					if indices is None:
						indices = raw.ch_names
					epochs = simplePipeline(raw, sample_rate=128, low_pass=75)
					epochs.save(file_path, overwrite=False)
				except Exception as e:
					failed_release_files.append((subject, e, file_path, traceback.format_exc()))
					print(f"Failed to process subject: {subject} in dataset release: {dataset_release}. Error: {e} Trace: {traceback.format_exc()}")
	gc.collect()


def _write_csv(df, filename):
	# if file does not exist write header 
	if not os.path.isfile(filename):
		df.to_csv(filename)
	else: # else it exists so append without writing the header
		df.to_csv(filename, mode='a', header=True)

'''
def main2(args):
	bids_path = BIDSPath(root=args.input_directory, datatype="eeg", suffix="eeg", extension=".set")

	subfolders = [f.path for f in os.scandir(args.input_directory) if f.is_dir() ]
	subfolders = subfolders[2:]
	subjects = [x.split("/")[-1][4:] for x in subfolders]
	print(f"Found {len(subjects)} subjects")

	indices = None
	means = pd.DataFrame()
	stds = pd.DataFrame()

	tasks = ["EC", "EO"]
	runs = [1, 2, 3, 4, 5]

	failed_files = []

	for subject in tqdm(subjects):
		print(f"Processing subject: {subject}")
		Path(os.path.join(args.output_dir, subject)).mkdir(parents=True, exist_ok=True)
		Path(os.path.join(args.output_dir, subject, "epochs")).mkdir(parents=True, exist_ok=True)
		bids_path.update(subject=subject)

		subject_all_data = []

		for task in tasks:
			bids_path.update(task=task)
			file_path = os.path.join(args.output_dir, subject, "epochs", f"{task}.fif")
			task_all_data = []
			if not os.path.exists(file_path):
				for run in runs:
					print(f"Processing subject: {subject} task {task} run {run}")
					try:
						bids_path.update(run=run)		
						raw = read_raw_bids(bids_path, extra_params={'preload':True}, verbose=False)
						raw.drop_channels(TO_DROP)
						mapping = {ch: 'eeg' for ch in raw.ch_names}
						raw.set_channel_types(mapping)
						raw.rename_channels(HBN_ELECTRODE_MAP_REVERSED)
						if indices is None:
							indices = raw.ch_names
						task_all_data.append(raw)
						subject_all_data.append(raw.get_data())
					except Exception as e:
						failed_files.append((subject, e, file_path, traceback.format_exc()))
						print(f"Failed to process subject: {subject} for task {task} on run {run}")
				concatenated_raw = mne.concatenate_raws(task_all_data)

				annotations_to_drop = [i for i in range(len(concatenated_raw.annotations))]
				concatenated_raw.annotations.delete(annotations_to_drop)

				epochs = simplePipeline(concatenated_raw)
				epochs.save(file_path, overwrite=False)

		if len(subject_all_data) > 0:
			subject_all_data = np.concatenate(subject_all_data, axis=1)

			# Z-score Transform By Channel
			mean = np.mean(subject_all_data, axis=1)
			std = np.std(subject_all_data, axis=1)

			means[subject] = mean
			stds[subject] = std

	if not means.empty:
		means.set_index(pd.Index(indices), inplace=True)
		_write_csv(means, os.path.join(args.output_dir, f"EOEC_means.csv"))
	if not stds.empty:
		stds.set_index(pd.Index(indices), inplace=True)
		_write_csv(stds, os.path.join(args.output_dir, f"EOEC_std.csv"))

	if failed_files:
		pd.DataFrame(failed_files, columns=["subject", "error", "file_path", "traceback"]).to_csv(
			os.path.join(args.output_dir, f"EOEC_failed_files.csv"), index=False
		)

	gc.collect()


def main3(args):
	dataset_releases = os.listdir(args.input_directory)

	patient_means = pd.DataFrame()
	patient_stds = pd.DataFrame()


	with open("missing_subjects.pkl", "rb") as f:
		missing_patients = pkl.load(f)

	missing_patients = set(missing_patients)

	for dataset_release in dataset_releases:
		print(f"Processing dataset release: {dataset_release}")
		_process_missing(dataset_release, args, patient_means, patient_stds, missing_patients)
		print(f"Finished processing dataset release: {dataset_release}, {patient_means.shape[1]} subjects processed")

	patient_means.to_csv(os.path.join(args.output_dir, "missing_patient_means.csv"))
	patient_stds.to_csv(os.path.join(args.output_dir, "missing_patient_stds.csv"))


	
def _process_missing(dataset_release, args, patient_means, patient_stds, missing_patients):
	bids_path = BIDSPath(root=os.path.join(args.input_directory, dataset_release),
					    datatype="eeg", suffix="eeg", extension=".set")
		
	subjects = set([bp.subject for bp in bids_path.match()])

	print(f"Found {len(subjects)} subjects for dataset release: {dataset_release}")

	missing_subjects_in_dataset = subjects.intersection(missing_patients)

	print(f"Found {len(missing_subjects_in_dataset)} subjects for dataset release: {dataset_release}")

	indices = None
	failed_files = []

	for subject in missing_subjects_in_dataset:
		print(f"Processing subject: {subject} in dataset release: {dataset_release}")
		subject_all_data = []
		bids_path.update(subject=subject)

		for bp in bids_path.match():
			try:		
				raw = read_raw_bids(bp, extra_params={'preload':True}, verbose=False)
				raw.drop_channels(TO_DROP)
				raw.rename_channels(HBN_ELECTRODE_MAP_REVERSED)
				if indices is None:
					indices = raw.ch_names
				subject_all_data.append(raw.get_data())
			except Exception as e:
				failed_files.append((subject, e, bp.task, traceback.format_exc()))
				print(f"Failed to process subject: {subject} in dataset release: {dataset_release}")

		if subject_all_data:
			subject_all_data = np.concatenate(subject_all_data, axis=1)

			# Z-score Transform By Channel
			mean = np.mean(subject_all_data, axis=1)
			std = np.std(subject_all_data, axis=1)

			patient_means[subject] = mean
			patient_stds[subject] = std

	if failed_files:
		pd.DataFrame(failed_files, columns=["subject", "error", "file_path", "traceback"]).to_csv(
			os.path.join(args.output_dir, f"missing_subjects_failed_files.csv"), index=False
		)

	gc.collect()

'''
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

	parser.add_argument(
		"--version", type=str, help="Version of the preprocessing"
	)

	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main(args)