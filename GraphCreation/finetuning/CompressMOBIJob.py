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

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
		for future in tqdm(futures):
			future.result()

def _process_subject(args, subject):
	original_subject_path = os.path.join(args.input_directory, subject)
	original_wavelet_path = os.path.join(original_subject_path, "wavelet_decompositions")
	original_graph_path = os.path.join(original_subject_path, "graphs")
	original_gait_path = os.path.join(original_subject_path, "gait")
	graph_file = os.listdir(original_graph_path)[0]
	if os.path.exists(original_wavelet_path) and os.path.exists(original_graph_path) and os.path.exists(original_gait_path):
		subject_path = os.path.join(args.output_dir, subject)
		Path(subject_path).mkdir(parents=True, exist_ok=True)
		for wavelet_file in os.listdir(original_wavelet_path):
			band = wavelet_file.split("_")[-2]
			epoch_file_path = os.path.join(original_wavelet_path, wavelet_file)
			if os.path.isfile(epoch_file_path) and band != "freq":
				epoch_data = torch.load(epoch_file_path)
				for i in range(epoch_data.shape[0]):
					epoch_path = os.path.join(subject_path, f"{wavelet_file[:-3]}_epoch_{i}")
					Path(epoch_path).mkdir(parents=True, exist_ok=True)
					curr_epoch = torch.tensor(epoch_data[i])
					curr_session = wavelet_file.split("_")[0]
					curr_gait = torch.load(os.path.join(original_gait_path, f"{curr_session}_epo_gait.pt"))
					curr_gait = curr_gait[i]
					graph = torch.load(os.path.join(original_graph_path, graph_file))

					torch.save(curr_epoch, os.path.join(epoch_path, f"{wavelet_file[:-3]}_epoch_{i}.pt"))
					torch.save(curr_gait, os.path.join(epoch_path, f"{curr_session}_epo_gait_epoch_{i}.pt"))
					torch.save(graph, os.path.join(epoch_path, f"{graph_file[:-3]}.pt"))

		
def parse_args():
	parser = argparse.ArgumentParser(description='Compress MOBI Wavelet Decompositions')
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