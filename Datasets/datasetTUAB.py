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
		folders = folders[:int(len(folders) * self.frac)]
		print(f"Loading {len(folders)} folders")
		with ThreadPoolExecutor(max_workers=16) as executor:
			futures = [executor.submit(self._process_folder, curr_folder, os.path.join(self.root, curr_folder, "wavelet_decompositions"), os.path.join(self.root, curr_folder, "graphs")) for curr_folder in folders]
			for future in tqdm(futures):
				future.result()
		self.length = len(self.epochs)

	def _process_folder(self, curr_folder, wavelet_folder, graph_folder):
		folder_epochs = dict()
		folder_graph_path = os.listdir(graph_folder)[0]
		split_path = folder_graph_path.split("_")
		folder_class = curr_folder.split("_")[-1]
		subject_int_class = 0 if folder_class == "normal" else 1
		graph = torch.load(os.path.join(graph_folder, folder_graph_path))
		graph.y = torch.tensor([subject_int_class])
		graph_name = "_".join(split_path[0:2])
		self.graphs[graph_name] = graph
		wavelet_files = os.listdir(wavelet_folder)
		for file_name in wavelet_files:
			attributes = file_name.split("_")
			epoch_idx = int(attributes[-1][:-3])
			band = attributes[-4]
			if epoch_idx not in folder_epochs:
				folder_epochs[epoch_idx] = dict()
				folder_epochs[epoch_idx]['graph_name'] = graph_name
			folder_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_folder, file_name))
		for epoch_idx, epoch_wavelet_dict in folder_epochs.items():
			epoch_wavelet_dict['delta'] = epoch_wavelet_dict['delta'][:, :240]
			epoch_wavelet_dict['theta'] = epoch_wavelet_dict['theta'][:, :240]
			epoch_wavelet_dict['alpha'] = epoch_wavelet_dict['alpha'][:, :480]
			epoch_wavelet_dict['beta'] = epoch_wavelet_dict['beta'][:, :960]
			epoch_wavelet_dict['gamma'] = epoch_wavelet_dict['gamma'][:, :1920]
			split_length_delta = epoch_wavelet_dict['delta'].shape[1] // 6
			split_length_theta = epoch_wavelet_dict['theta'].shape[1] // 6
			split_length_beta = epoch_wavelet_dict['beta'].shape[1] // 6
			split_length_alpha = epoch_wavelet_dict['alpha'].shape[1] // 6
			split_length_gamma = epoch_wavelet_dict['gamma'].shape[1] // 6
			for split_idx in range(6):
				epoch_tuple = (epoch_wavelet_dict['graph_name'], wavelet_folder, (epoch_idx, split_idx),
				epoch_wavelet_dict['delta'][:, split_length_delta*split_idx: split_length_delta*(split_idx + 1)],
				epoch_wavelet_dict['theta'][:, split_length_theta*split_idx: split_length_theta*(split_idx + 1)], 
				epoch_wavelet_dict['alpha'][:, split_length_alpha*split_idx: split_length_alpha*(split_idx + 1)],
				epoch_wavelet_dict['beta'][:,  split_length_beta*split_idx:  split_length_beta*(split_idx  + 1)],
				epoch_wavelet_dict['gamma'][:, split_length_gamma*split_idx: split_length_gamma*(split_idx + 1)])
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
	train_dataset = WaveletTUABDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUAB-128Hz/train", frac=1.0)
	eval_dataset = WaveletTUABDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUAB-128Hz/eval", frac=1.0)
	print("Length of train dataset: ", len(train_dataset))
	print("Length of val dataset: ", len(eval_dataset))
	data = train_dataset[0]
	print("Data: ", len(data), data['graph'], data['wavelet_folder'], data['delta'].shape, data['gamma'].shape, "Data Label:", data['graph'].y)

	print("Train Graphs: ", len(train_dataset.graphs)) # Should be 2717, One for each folder
	print("Train Epochs: ", len(train_dataset.epochs)) # Should be 61223

	print("Val Graphs: ", len(eval_dataset.graphs)) # Should be 276
	print("Val Epochs: ", len(eval_dataset.epochs)) # Should be 6067

	from torch_geometric.loader import DataLoader
	finetune_train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)
	finetune_eval_loader = DataLoader(eval_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)

	data = next(iter(finetune_train_loader))
	edge_index = data['graph'].edge_index
	edge_dist = data['graph'].edge_attr
	assert edge_index[0].min() == 0 and edge_index[0].max() <= 76, f"{edge_index[0].min()}, {edge_index[0].max()}"
	assert edge_index[1].min() == 0 and edge_index[1].max() <= 76, f"{edge_index[1].min()}, {edge_index[1].max()}"
	assert edge_dist.min() == 0 and edge_dist.max() <= 1

	data = next(iter(finetune_eval_loader))
	edge_index = data['graph'].edge_index
	edge_dist = data['graph'].edge_attr
	assert edge_index[0].min() == 0 and edge_index[0].max() <= 76, f"{edge_index[0].min()}, {edge_index[0].max()}"
	assert edge_index[1].min() == 0 and edge_index[1].max() <= 76, f"{edge_index[1].min()}, {edge_index[1].max()}"
	assert edge_dist.min() == 0 and edge_dist.max() <= 1