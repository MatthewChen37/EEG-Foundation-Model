import torch
import os, argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def main(args):
	subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
	print("Subjects: ", len(subjects))

	subjects_grouped = group_list(subjects, 32)
	print(f"Processing {len(subjects_grouped)} groups")

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject_group, args, subject_group) for subject_group in subjects_grouped]
		for future in tqdm(futures):
			future.result()

def _process_subject_group(args, subjects):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
		for future in futures:
			future.result()

def _process_subject(args, subject):
	original_subject_path = os.path.join(args.input_directory, subject)
	original_wavelet_path = os.path.join(original_subject_path, "wavelet_decompositions")
	original_graph_path = os.path.join(original_subject_path, "graphs_v2")
	if os.path.exists(original_wavelet_path) and os.path.exists(original_graph_path):
		subject_path = os.path.join(args.output_dir, subject)
		wavelet_path = os.path.join(subject_path, "wavelet_decompositions")
		graph_path = os.path.join(subject_path, "graphs")
		Path(subject_path).mkdir(parents=True, exist_ok=True)
		Path(wavelet_path).mkdir(parents=True, exist_ok=True)
		Path(graph_path).mkdir(parents=True, exist_ok=True)

		for wavelet_file in os.listdir(original_wavelet_path):
			band = wavelet_file.split("_")[-2]
			epoch_file_path = os.path.join(original_wavelet_path, wavelet_file)
			if os.path.isfile(epoch_file_path) and band != "freq":
				epoch_data = torch.load(epoch_file_path).data
				if epoch_data.shape[0] > 10 and epoch_data.shape[0] < 50:
					epoch_data = epoch_data[10:] # Add additional samples
					for i in range(epoch_data.shape[0]):
						curr_epoch = torch.tensor(epoch_data[i])
						torch.save(curr_epoch, os.path.join(wavelet_path, f"{wavelet_file[:-3]}_epoch_{10 + i}.pt"))

		for graph_file in os.listdir(original_graph_path):
			epoch_file_path = os.path.join(original_graph_path, graph_file)
			if os.path.isfile(epoch_file_path):
				epoch_data = torch.load(epoch_file_path)
				torch.save(epoch_data, os.path.join(graph_path, f"{graph_file[:-3]}.pt"))

def parse_args():
	parser = argparse.ArgumentParser(description='Compress TUH Wavelet Decompositions')
	parser.add_argument('--input_directory', type=str, required=True,
					  help='Path to BIDS directory containing the preprocessed data')
	parser.add_argument('--output_dir', type=str, required=True,
					  help='Processing will write to this directory.')
	args = parser.parse_args()

	Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	return args

if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	main(args)