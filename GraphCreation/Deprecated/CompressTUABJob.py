import torch
import os, argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch_geometric.data import Data

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

failed_files = []
def main(args):
	files = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
	print("Files: ", len(files))

	with ProcessPoolExecutor() as executor:
		files_grouped = group_list(files, 8)
		futures = [executor.submit(_process_file, args, files) for files in files_grouped]
		for future in tqdm(futures):
			future.result()

def _process_file(args, files):
	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_file_thread, args, file) for file in files]
		for future in  futures:
			result = future.result()
			if result is not None:
				failed_files.append(result)
	return None

def _process_file_thread(args, file_name):
	original_file_path = os.path.join(args.input_directory, file_name)
	original_wavelet_path = os.path.join(original_file_path, "wavelet_decompositions")
	original_graph_path = os.path.join(original_file_path, "graphs_v2")

	if os.path.exists(original_wavelet_path) and os.path.exists(original_graph_path):
		subject = file_name.split("_")[0]
		subject_class = file_name.split("-")[-1]
		subject_path = os.path.join(args.output_dir, f"{subject}-{subject_class}")
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
				for i in range(epoch_data.shape[0]):
					curr_epoch = torch.tensor(epoch_data[i])
					torch.save(curr_epoch, os.path.join(wavelet_path, f"{wavelet_file[:-3]}_epoch_{i}.pt"))
		for graph_file in os.listdir(original_graph_path):
			epoch_file_path = os.path.join(original_graph_path, graph_file)
			if os.path.isfile(epoch_file_path):
				epoch_data = torch.load(epoch_file_path)
				subject_int_class = 0 if subject_class == "normal" else 1
				epoch_data.y = torch.tensor([subject_int_class])
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