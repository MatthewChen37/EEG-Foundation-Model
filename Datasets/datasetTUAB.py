import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletTUABDataset(Dataset):
	def __init__(self, root, frac=1.0, transform=None):
		super(WaveletTUABDataset, self).__init__(root, transform)

		self.frac = frac

		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

	def _index_data(self):
		self.graphs = dict()
		self.epochs = []
		folders = os.listdir(self.root)
		print(f"Loading {len(folders)} folders")
		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(self._process_folder, os.path.join(self.root, curr_folder, "wavelet_decompositions"), os.path.join(self.root, curr_folder, "graphs")) for curr_folder in folders]
			for future in tqdm(futures):
				future.result()
		self.length = len(self.epochs)

	def _process_folder(self, wavelet_folder, graph_folder):
		folder_epochs = dict()
		folder_graph_path = os.listdir(graph_folder)[0]
		self.graphs[folder_graph_path.split("_")[0]] = torch.load(os.path.join(graph_folder, folder_graph_path))
		wavelet_files = os.listdir(wavelet_folder)
		for file_name in wavelet_files:
			attributes = file_name.split("_")
			epoch_idx = int(attributes[-1][:-3])
			band = attributes[-4]
			if epoch_idx not in folder_epochs:
				folder_epochs[epoch_idx] = dict()
				folder_epochs[epoch_idx]['graph_name'] = attributes[0]
			folder_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_folder, file_name))
		for epoch_idx, epoch_wavelet_dict in folder_epochs.items():
			epoch_tuple = (epoch_wavelet_dict['graph_name'], wavelet_folder, epoch_idx,
				epoch_wavelet_dict['delta'], epoch_wavelet_dict['theta'], epoch_wavelet_dict['alpha'], epoch_wavelet_dict['beta'],
				epoch_wavelet_dict['gamma'])
			self.epochs.append(epoch_tuple)

	def len(self):
		return self.length

	def get(self, idx):
		epoch_tuple = self.epochs[idx]
		graph = self.graphs[epoch_tuple[0]]
		data = {
			"graph": graph, # Labels should be stored in the graph.y 
			"wavelet_folder": epoch_tuple[1],
			"delta": epoch_tuple[3],
			"theta": epoch_tuple[4],
			"alpha": epoch_tuple[5],
			"beta": epoch_tuple[6],
			"gamma": epoch_tuple[7],
		}
		return data

		
if __name__ == "__main__":

	# dataset = EEGDataset(root="/home/hice1/mchen439/data/TUH-Processed", frac=0.0001)

	dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.01)

	#dataset = WaveletFinetuningDataset(root="/home/hice1/mchen439/scratch/downstreamTaskData/eval")

	print("Length of dataset: ", len(dataset))

	data = dataset[0]

	#print("Data: ", len(data), data['graph'], data['wavelet_folder'], data['delta'].shape, data['gamma'].shape, "Data Label:", data['graph'].y)

	print("Data: ", len(data), data['graph'], data['subject_name'], data['delta'].shape, data['gamma'].shape, "Data Label:", data['graph'].y)