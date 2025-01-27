import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm


BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class Subject:
	def __init__(self, subject_path, subject_graph_path, subject_wavelet_file_names):
		self.subject_path = subject_path
		self.subject_graph_path = subject_graph_path
		self.subject_wavelet_file_names = subject_wavelet_file_names

	def __len__(self):
		return len(self.file_names)
	
	def getSubjectPath(self):
		return self.subject_path
	
	def getSubjectGraphPath(self):
		return self.subject_graph_path
	
	def getSubjectWaveletFileNames(self):
		return self.subject_wavelet_file_names
	
	def __str__(self):
		subject_string = f"Subject: {self.subject_path}, Graphs: {self.subject_graph_path}"

		for band, file_name in self.subject_wavelet_file_names.items():
			subject_string += f"\n {band}: {torch.load(file_name[0]).node_name}"
		return subject_string


class WaveletDataset(Dataset):
	def __init__(self, root, frac=1.0, transform=None):
		super(WaveletDataset, self).__init__(root, transform)
		self.frac = frac

		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

	def _index_data(self):
		self.graphs = dict()
		self.epochs = []

		subjects = [f.path.split("/")[-1] for f in os.scandir(self.root) if f.is_dir()]
		subjects = subjects[:int(len(subjects) * self.frac)]
		print(f"Loading {len(subjects)} subjects")
		for subject in tqdm(subjects):
			subject_graph_path = os.path.join(self.root, subject, "graphs")
			if os.path.exists(subject_graph_path) and len(os.listdir(subject_graph_path)) > 0:
				subject_graph_path = os.listdir(subject_graph_path)[0]
				self.graphs[subject_graph_path.split("_")[0]] = torch.load(os.path.join(self.root, subject, "graphs_v2", subject_graph_path))
				subject_path = os.path.join(self.root, subject)
				wavelet_files = os.listdir(os.path.join(subject_path, "wavelet_decompositions"))
				subject_epochs = dict()
				for file_name in wavelet_files:
					epoch_idx = int(attributes[-1][:-3])
					band = attributes[-4]
					if epoch_idx not in subject_epochs:
						subject_epochs[epoch_idx] = dict()
						subject_epochs[epoch_idx]['graph_name'] = attributes[0] + "_" + attributes[1]
					subject_epochs[epoch_idx][band] = torch.load(file_name)
				for epoch_idx, epoch_wavelet_dict in subject_epochs.items():
					epoch_tuple = (epoch_wavelet_dict['graph_name'], subject, epoch_idx, epoch_wavelet_dict['delta'],
								   epoch_wavelet_dict['theta'], epoch_wavelet_dict['alpha'],
								   epoch_wavelet_dict['beta'], epoch_wavelet_dict['gamma'])
					self.epochs.append(epoch_tuple)
		self.length = len(self.epochs)

	def len(self):
		return self.length
	
	def get(self, idx):
		epoch_tuple = self.epochs[idx]
		graph = self.graphs[epoch_tuple[0]]
		data = {
			"graph": graph,
			"subject_name": epoch_tuple[1],
			"delta": epoch_tuple[3],
			"theta": epoch_tuple[4],
			"alpha": epoch_tuple[5],
			"beta": epoch_tuple[6],
			"gamma": epoch_tuple[7],
		}

class EEGDataset(Dataset):
	def __init__(self, root, frac=1.0, transform=None):
		super(EEGDataset, self).__init__(root, transform)
		self.frac = frac

		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

	def _index_data(self):
		self.file_names = []
		subjects = [f.path.split("/")[-1] for f in os.scandir(self.root) if f.is_dir()]
		subjects = subjects[:int(len(subjects) * self.frac)]
		print(f"Loading {len(subjects)} subjects")
		for subject in tqdm(subjects):
			subject_path = os.path.join(self.root, subject, "graphs")
			subject_file_names = os.listdir(subject_path)
			valid_files = []
			for file_name in subject_file_names:
				if not file_name.endswith("_0.pt"):
					valid_files.append(os.path.join(subject_path, file_name))
			self.file_names.extend(valid_files)
		self.length = len(self.file_names)
	
	def len(self):
		return self.length
	
	def get(self, idx):
		data = torch.load(os.path.join(self.file_names[idx]))
		data.x = data.x * 1000
		return data
	
		
if __name__ == "__main__":

	# dataset = EEGDataset(root="/home/hice1/mchen439/data/TUH-Processed", frac=0.0001)

	dataset = WaveletDataset(root="/home/hice1/mchen439/data/TUH-Processed", frac=0.0001)

	print("Length of dataset: ", len(dataset))

	print("Number of subjects: ", len(dataset.subjects))

	data = dataset[0]

	print("Data: ", data)