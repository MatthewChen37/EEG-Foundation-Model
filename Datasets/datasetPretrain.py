import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletPretrainDataset(Dataset):
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

		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(self._process_subject, subject) for subject in subjects]
			for future in tqdm(futures):
				future.result()

		self.length = len(self.epochs)
			

	def _process_subject(self, subject):
		subject_graph_folder = os.path.join(self.root, subject, "graphs")
		wavelet_path = os.path.join(self.root, subject, "wavelet_decompositions")
		if os.path.exists(subject_graph_folder) and len(os.listdir(subject_graph_folder)) > 0 and os.path.exists(wavelet_path) and len(os.listdir(wavelet_path)) > 0:
			subject_graph_path = os.listdir(subject_graph_folder)[0]
			self.graphs[subject_graph_path.split("_")[0]] = torch.load(os.path.join(subject_graph_folder, subject_graph_path), weights_only=False)
			wavelet_files = os.listdir(wavelet_path)
			subject_epochs = dict()
			for file_name in wavelet_files:
				attributes = file_name.split("_")
				epoch_idx = int(attributes[-1][:-3])
				band = attributes[-4]
				if epoch_idx not in subject_epochs:
					subject_epochs[epoch_idx] = dict()
					subject_epochs[epoch_idx]['graph_name'] = attributes[0]
				subject_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_path, file_name), weights_only=False)
			for epoch_idx, epoch_wavelet_dict in subject_epochs.items():
				epoch_tuple = (epoch_wavelet_dict['graph_name'], subject, epoch_idx, epoch_wavelet_dict['delta'],
								epoch_wavelet_dict['theta'], epoch_wavelet_dict['alpha'],
								epoch_wavelet_dict['beta'], epoch_wavelet_dict['gamma'])
				self.epochs.append(epoch_tuple)

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
		return data


if __name__ == "__main__":
	dataset = WaveletDataset(root="/home/hice1/mchen439/scratch/eegfoundationmodeldata", frac=0.01)

	print("Length of dataset: ", len(dataset))

	data = dataset[0]

	#print("Data: ", len(data), data['graph'], data['wavelet_folder'], data['delta'].shape, data['gamma'].shape, "Data Label:", data['graph'].y)

	print("Data: ", len(data), data['graph'], data['subject_name'], data['delta'].shape, data['gamma'].shape, "Data Label:", data['graph'].y)