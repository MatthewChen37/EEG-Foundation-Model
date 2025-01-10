import torch
from torch_geometric.data import Dataset

class EEGDataset(Dataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(EEGDataset, self).__init__(root, transform, pre_transform)

		self.length = 


