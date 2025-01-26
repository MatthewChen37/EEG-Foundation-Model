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

class WaveletDataset(Dataset):
	def __init__(self, root, frac=1.0, transform=None):
		super(WaveletDataset, self).__init__(root, transform)
		self.frac = frac

		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

	def _index_data(self):
		self.subjects = []

		subjects = [f.path.split("/")[-1] for f in os.scandir(self.root) if f.is_dir()]
		subjects = subjects[:int(len(subjects) * self.frac)]
		print(f"Loading {len(subjects)} subjects")
		for subject in tqdm(subjects):
			subject_graph_path = os.listdir(os.path.join(self.root, subject, "graphs_v2"))[0]
			subject_graph_path = os.path.join(self.root, subject, "graphs_v2", subject_graph_path)
			subject_file_names = {
				"delta": [],
				"theta": [],
				"alpha": [],
				"beta": [],
				"gamma": []
			}
			subject_path = os.path.join(self.root, subject)
			wavelet_files = os.listdir(os.path.join(subject_path, "wavelet_decompositions"))
			for file_name in wavelet_files:
				band = file_name.split("_")[-2]
				subject_file_names[band].append(os.path.join(subject_path, "wavelet_decompositions", file_name))
			self.subjects.append(Subject(subject_path, subject_graph_path, subject_file_names))
		self.length = len(self.subjects)

	def len(self):
		return self.length
	
	def get(self, idx):
		return self.subjects[idx]

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

	dataset = EEGDataset(root="/home/hice1/mchen439/data/TUH-Processed", frac=0.0001)

	print("Length of dataset: ", len(dataset))

	print("Length of file names: ", len(dataset.file_names))

	data = dataset[0]

	print("Data: ", data)