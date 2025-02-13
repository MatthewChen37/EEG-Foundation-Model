import argparse
from pathlib import Path
import warnings
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def main(args):
	subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
	print("Subjects: ", len(subjects))

	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_delete_folder, args, subject) for subject in subjects]
		for future in tqdm(futures):
			future.result()

def _delete_folder(args, subject):
	original_subject_path = os.path.join(args.input_directory, subject)
	original_wavelet_path = os.path.join(original_subject_path, "wavelet_decompositions")
	original_graph_path = os.path.join(original_subject_path, "graphs_v2")
	if os.path.exists(original_wavelet_path) and os.path.exists(original_graph_path):
		shutil.rmtree(original_wavelet_path)
		shutil.rmtree(original_graph_path)

def parse_args():
	parser = argparse.ArgumentParser(description='Preprocess TUH EEG data')
	parser.add_argument('--input_directory', type=str, help='Path to the directory containing the TUH EEG data')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Ignore warnings
	warnings.filterwarnings("ignore")

	args = parse_args()
	main(args)

