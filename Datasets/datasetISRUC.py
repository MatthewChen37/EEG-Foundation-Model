import torch
import os
import warnings
import numpy as np
import math
import mne
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from einops import rearrange
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletISRUCDataset(Dataset):
	def __init__(self, root, include_high=False, frac=1.0, transform=None):
		super(WaveletISRUCDataset, self).__init__(root, transform)
		self.frac = frac
		self.include_high = include_high
		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

		self.og_indices = [
			2,
			4,
			8,
			3,
			5,
			9,
		]

	def _index_data(self):
		self.graphs = dict()
		self.epochs = []
		folders = os.listdir(self.root)
		folders = folders[:int(len(folders) * self.frac)]
		print(f"Loading {len(folders)} folders")
		with ThreadPoolExecutor(max_workers=64) as executor:
			futures = [executor.submit(self._process_folder, os.path.join(self.root, curr_folder, "wavelet_decompositions"), os.path.join(self.root, curr_folder, "graphs")) for curr_folder in folders]
			for future in tqdm(futures):
				future.result()
		self.length = len(self.epochs)

	def _process_folder(self, wavelet_folder, graph_folder):
		folder_graph_path = os.listdir(graph_folder)[0]
		split_path = folder_graph_path.split("_")
		graph = torch.load(os.path.join(graph_folder, folder_graph_path)).clone()
		graph.x = None
		graph_name = "_".join(split_path[:5])
		self.graphs[graph_name] = graph
		
		# Store file paths instead of loading the data
		wavelet_files = os.listdir(wavelet_folder)
		band_file_paths = {}
		for file_name in wavelet_files:
			attributes = file_name.split("_")
			band = attributes[-4]
			if band != "high":
				band_file_paths[band] = os.path.join(wavelet_folder, file_name)
		
		# Store only the graph name, wavelet folder, and file paths for lazy loading
		epoch_tuple = (graph_name, wavelet_folder, band_file_paths)
		self.epochs.append(epoch_tuple)

	def len(self):
		return self.length
	
	def get(self, idx):
		epoch_tuple = self.epochs[idx]
		graph_main = torch.load("/home/hice1/mchen439/scratch/TUEV_small/eval/1_spsw_005_a__event_0/graphs/1_spsw_005_a__event_0_graph_epoch_0.pt")
		graph = self.graphs[epoch_tuple[0]]

		band_file_paths = epoch_tuple[2]

		# Load wavelet data lazily on-demand
		data = {
			'graph': graph,
			'wavelet_folder': epoch_tuple[1],
			'delta': torch.load(band_file_paths['delta']),
			'theta': torch.load(band_file_paths['theta']),
			'alpha': torch.load(band_file_paths['alpha']),
			'beta': torch.load(band_file_paths['beta']),
			'gamma': torch.load(band_file_paths['gamma']),
		}

		new_data = {
			'delta': torch.zeros(data['delta'].shape[0], 19, data['delta'].shape[2], dtype=torch.double),
			'theta': torch.zeros(data['theta'].shape[0], 19, data['theta'].shape[2], dtype=torch.double),
			'alpha': torch.zeros(data['alpha'].shape[0], 19, data['alpha'].shape[2], dtype=torch.double),
			'beta': torch.zeros(data['beta'].shape[0], 19, data['beta'].shape[2], dtype=torch.double),
			'gamma': torch.zeros(data['gamma'].shape[0], 19, data['gamma'].shape[2], dtype=torch.double),
		}

		new_data['delta'][:, self.og_indices, :] = data['delta']
		new_data['theta'][:, self.og_indices, :] = data['theta']
		new_data['alpha'][:, self.og_indices, :] = data['alpha']
		new_data['beta'][:, self.og_indices,  :] = data['beta']
		new_data['gamma'][:, self.og_indices, :] = data['gamma']

		new_data['delta'] = new_data['delta'][:, :, 3:123]
		new_data['theta'] = new_data['theta'][:, :, 3:123]
		new_data['alpha'] = new_data['alpha'][:, :, 3:243]
		new_data['beta'] = new_data['beta'][:, :,   3:483]
		new_data['gamma'] = new_data['gamma'][:, :, 2:962]

		#print("MAIN", graph_main.edge_index.shape, graph_main.edge_attr.shape)
		graph.edge_attr = graph_main.edge_attr.unsqueeze(0).repeat(new_data['delta'].shape[0], 1, 1)
		graph.edge_attr = graph.edge_attr.reshape(-1, graph.edge_attr.shape[-1])

		graph.edge_index = graph_main.edge_index.unsqueeze(0).repeat(new_data['delta'].shape[0], 1, 1)
		#print(graph.edge_index.shape, graph_main.num_nodes)
		batch_offset = 19 * torch.arange(new_data['delta'].shape[0], dtype=torch.long).unsqueeze(1).repeat(1, 2).unsqueeze(2).repeat(1, 1, graph.edge_index.shape[-1]).to(graph.edge_index.device)
		#print(batch_offset, batch_offset.shape)
		graph.edge_index = graph.edge_index + batch_offset
		#print(graph.edge_index.min(), graph.edge_index.max())
		graph.edge_index = rearrange(graph.edge_index, 'b n m -> n (b m)')
		#graph.edge_index = graph.edge_index.reshape(-1, graph.edge_index.shape[-1])
		#print(graph.edge_index.shape, graph.edge_attr.shape)
		graph.pos = graph_main.pos.unsqueeze(0).repeat(new_data['delta'].shape[0], 1, 1)
		graph.pos = graph.pos.reshape(-1, graph.pos.shape[-1])

		#print("FINAL", graph.pos.shape, graph.edge_index.shape, graph.edge_attr.shape, graph.y.shape)
		new_data['graph'] = graph
		new_data['wavelet_folder'] = epoch_tuple[1]
		return new_data
	
if __name__ == "__main__":
	train_dataset = WaveletISRUCDataset(root="../../ISRUC_Best/train", frac=1.0)
	eval_dataset = WaveletISRUCDataset(root="../../ISRUC_Best/test", frac=1.0)

	print("Length of train dataset: ", len(train_dataset))
	print("Length of val dataset: ", len(eval_dataset))

	data = train_dataset[0]
	print("Data: ", len(data), data['graph'], data['wavelet_folder'],
                    data['delta'].shape,
                    data['theta'].shape,
                    data['alpha'].shape,
                    data['beta'].shape,
                    data['gamma'].shape,
                    "Data Label:", data['graph'].y,
                    data['graph'].edge_index.shape,
                    data['graph'].edge_attr.shape)
	
	edge_index = data['graph'].edge_index
	edge_dist = data['graph'].edge_attr

	print("Train Graphs: ", len(train_dataset.graphs)) 
	print("Train Epochs: ", len(train_dataset.epochs))

	print("Val Graphs: ", len(eval_dataset.graphs))
	print("Val Epochs: ", len(eval_dataset.epochs))

	from torch_geometric.loader import DataLoader
	finetune_train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)
	finetune_eval_loader = DataLoader(eval_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)
	print(len(finetune_train_loader), len(finetune_eval_loader))

	data = next(iter(finetune_train_loader))
	edge_index = data['graph'].edge_index
	edge_dist = data['graph'].edge_attr
	assert edge_index[0].min() == 0 and edge_index[0].max() <= (19 * 20 * 4) - 1, f"{edge_index[0].min()}, {edge_index[0].max()}"
	assert edge_index[1].min() == 0 and edge_index[1].max() <= (19 * 20 * 4) - 1, f"{edge_index[1].min()}, {edge_index[1].max()}"
	assert edge_dist.min() == 0 and edge_dist.max() <= 1
	print("Data: ", len(data), data['graph'], data['wavelet_folder'],
                    data['delta'].shape,
                    data['theta'].shape,
                    data['alpha'].shape,
                    data['beta'].shape,
                    data['gamma'].shape,
                    "Data Label:", data['graph'].y.shape,
                    data['graph'].edge_index.shape,
                    data['graph'].edge_attr.shape)

	data = next(iter(finetune_eval_loader))
	edge_index = data['graph'].edge_index
	edge_dist = data['graph'].edge_attr
	assert edge_index[0].min() == 0 and edge_index[0].max() == (19 * 20 * 4) - 1, f"{edge_index[0].min()}, {edge_index[0].max()}"
	assert edge_index[1].min() == 0 and edge_index[1].max() == (19 * 20 * 4) - 1, f"{edge_index[1].min()}, {edge_index[1].max()}"
	assert edge_dist.min() == 0 and edge_dist.max() <= 1
	print("Data: ", len(data), data['graph'], data['wavelet_folder'],
                    data['delta'].shape,
                    data['theta'].shape,
                    data['alpha'].shape,
                    data['beta'].shape,
                    data['gamma'].shape,
                    "Data Label:", data['graph'].y.shape,
                    data['graph'].edge_index.shape,
                    data['graph'].edge_attr.shape)



	'''
	torch.set_printoptions(profile="full")
	print(edge_index)
	print("---------------")
	print(edge_dist)
	'''

	print("All tests passed")
