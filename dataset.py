import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm

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