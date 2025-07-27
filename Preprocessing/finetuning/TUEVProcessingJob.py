import mne
import sys
sys.path.append("../")
import pandas as pd
import os, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from preprocessingPipeline import simplePipeline, group_list, simplePipelineNoEpoch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import traceback
from convertTUHtoBIDS import _rename_channels, CHANNELS_TO_KEEP

def main(args):
	sessions = [f.path for f in os.scandir(args.input_directory) if f.is_dir()]
	print(f"Found {len(sessions)} session")
	session_grouped = group_list(sessions, 10)
	print(f"Processing {len(session_grouped)} groups")
	# Define the columns for the empty DataFrame
	columns = ['f_path', 'error', 'traceback']

	failed_files = []
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_file_group, args, session_group) for session_group in session_grouped]
		for future in tqdm(futures):
			result = future.result()
			failed_files.extend(result)

	failed_files_df = pd.DataFrame(failed_files, columns=columns)
	print(f"Failed files: {len(failed_files)}")
	failed_files_df.to_csv(os.path.join(args.output_dir, "failed_files.csv"), index=False)

def _process_file_group(args, session_group):
	failed_files = []
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(process_session, os.path.join(args.input_directory, session)) for session in session_group]
		for future in futures:
			result = future.result()
			if result is not None:
				failed_files.append(result)

	return failed_files

# From https://github.com/fjssharpsword/MedIR/blob/3d1eef266e8ad82a0cfad7a6111076028b8d42fa/EEG/TUSZ/dsts/tuev_spsw.py#L87
def parse_annotation(ann_path):
	# Annotation in one bipolar channel means annotation in both
	montage = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-Cz',\
                   'Cz-C4', 'C4-T4', 'T4-A2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
	annotations = []
	with open(ann_path, 'r') as ann_file:
		for line in ann_file.readlines():
			line = line.split(',')
			ch, st, ed, cl =  eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3])
			ch_names = montage[ch].split('-')
			channel_1 = ch_names[0]
			channel_2 = ch_names[1]
			annotations.append(([channel_1, channel_2], st, ed, cl))
	return annotations

def process_session(session_path):
	file_list = os.listdir(session_path)
	for file in file_list:
		if file.endswith('.edf'):
			try: 
				raw = mne.io.read_raw_edf(os.path.join(session_path, file), preload=True, verbose=False)
				_rename_channels(raw)
				raw.info['line_freq'] = 60
				annotations_file = file[:-4] + '.rec'
				annotations_list = parse_annotation(os.path.join(session_path, annotations_file))
				annotations_df = pd.DataFrame(annotations_list, columns=['channel', 'start', 'end', 'class'])
				annotations_df['duration'] = annotations_df['end'] - annotations_df['start']
				annotations = mne.Annotations(onset=annotations_df['start'], duration=annotations_df['duration'], description=annotations_df['class'], ch_names=annotations_df['channel'].tolist())

				raw.set_annotations(annotations)
				raw.set_montage("standard_1005", on_missing="ignore")
				raw.set_meas_date(None)

				to_drop = [ch for ch in raw.ch_names if ch not in CHANNELS_TO_KEEP]
				raw.drop_channels(to_drop)
				if 'A1' and 'A2' in raw.ch_names:
					raw = raw.drop_channels(['A1', 'A2'])
				assert len(raw.ch_names) == 19, f"Number of channels is {len(raw.ch_names)}"

				raw = simplePipelineNoEpoch(raw, sample_rate=128, low_pass=75)
				for idx, annotation in enumerate(annotations):
					raw_annotation_event = build_events(raw.copy(), annotation)
					if raw_annotation_event.times[-1] - raw_annotation_event.times[0] != 6:
						print(f"Annotation {annotation} is not 20 seconds long ------------------------------------")
					if args.split == "train":
						file_ouput_path = os.path.join(args.output_dir, f"{annotation['description']}_{file[:-4]}_event_{idx}.fif")
					else:
						# Turns out the eval impromperly labels the classes, i.e. there are artf and eyem 
						# labels inside backg .rec files - can't trust shit
						file_ouput_path = os.path.join(args.output_dir, f"{annotation['description']}_{file[:-4]}_event_{idx}.fif")
					if os.path.exists(file_ouput_path):
						continue
					else:
						raw_annotation_event.save(file_ouput_path, overwrite=False)
			except Exception as e:
				print(f"Failed to process {file, session_path}, error: {e} {traceback.format_exc()}")
				return (os.path.join(file, session_path), e, traceback.format_exc())
		else:
			pass # ignore other files

def build_events(raw, annotation):
	offset = raw.times[-1]
	raw_modified = mne.concatenate_raws([raw, raw, raw])
	raw_annotation_event = raw_modified.copy().crop(offset + annotation['onset'] - 2, offset + annotation['onset'] + round(annotation['duration']) + 3)
	return raw_annotation_event

def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess TUEV data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the TUEV data')
	parser.add_argument('--output_dir', type=str, help='Path to the directory where the preprocessed data will be stored')
	parser.add_argument('--split', type=str, default='train')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)

