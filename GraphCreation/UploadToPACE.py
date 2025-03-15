import shutil
import torch
import os, argparse
import numpy as np
import pandas as pd
import warnings
import traceback
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch_geometric.data import Data

def group_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def main(args):
	subjects = [f.path.split("/")[-1] for f in os.scandir(args.input_directory) if f.is_dir()]
	print("Subjects: ", len(subjects))
	subject_groups = group_list(subjects, 30)
	print("Subject Groups: ", len(subject_groups))
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(_process_subject_group, args, subject_group) for subject_group in subject_groups]
		for future in tqdm(futures):
			future.result()

def _process_subject_group(args, subjects):

	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(_process_subject, args, subject) for subject in subjects]
		for future in futures:
			future.result()

def _process_subject(args, subject):
	base_path = os.path.join(args.input_directory, subject)
	wavelet_path = os.path.join(base_path, f"wavelet_decompositions_v{args.version}")
	graph_path = os.path.join(base_path, f"graphs_v{args.version}")

	output_subject_path = os.path.join(args.output_directory, subject)
	output_subject_wavelet_path = os.path.join(output_subject_path, f"wavelet_decompositions_v{args.version}")
	output_subject_graph_path = os.path.join(output_subject_path, f"graphs_v{args.version}")

	if os.path.exists(wavelet_path) and os.path.exists(graph_path):
		Path(output_subject_path).mkdir(parents=True, exist_ok=True)
		Path(output_subject_wavelet_path).mkdir(parents=True, exist_ok=True)
		Path(output_subject_graph_path).mkdir(parents=True, exist_ok=True)	

		for wavelet_file in os.listdir(wavelet_path):
			if not os.path.isfile(os.path.join(output_subject_wavelet_path, wavelet_file)):
				shutil.copy(os.path.join(wavelet_path, wavelet_file), os.path.join(output_subject_wavelet_path, wavelet_file))

		for graph_file in os.listdir(graph_path):
			if not os.path.isfile(os.path.join(output_subject_graph_path, graph_file)):
				graph = torch.load(os.path.join(graph_path, graph_file))
				graph.x = None
				torch.save(graph, os.path.join(output_subject_graph_path, graph_file))

def parse_args():
    parser = argparse.ArgumentParser(description='Selectively create data for uploading to PACE')
    parser.add_argument('--input_directory', type=str, required=True,
					  help='Input directory containing the preprocessed data')
    parser.add_argument('--output_directory', type=str, required=True, help='Output directory for the graphs')
    parser.add_argument('--version', type=str, help='Version of the preprocessing')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	args = parse_args()
	main(args)