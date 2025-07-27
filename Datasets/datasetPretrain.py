"""
import torch
import os
import warnings
from torch.utils.data import ConcatDataset
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class WaveletPretrainDataset(Dataset):
	def __init__(self, root, include_high=False, frac=1.0, suffix="v128Hz", transform=None):
		super().__init__(root, transform)
		self.frac = frac
		self.suffix = suffix
		self.include_high = include_high
		'''
		Data segments for pretraining are 60 seconds long
		For 128Hz, this should mean the original EEG data length is 128 * 60 = 7680 time points,
		which would imply that wavelet lengths are 7680 / 2^N  long for each level. However,
		https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html#single-level-dwt
		in preprocessing we use the default symmetric mode. Thus, series lengths are: 
		len(cA) == len(cD) == floor((len(data) + wavelet.dec_len - 1) / 2) long. One can set: 
		mode to "periodization" to a better match:
		len(cA) == len(cD) == ceil(len(data) / 2).

		Here we force the length to be "nice" by truncating regardless of the mode.
		'''
		if self.suffix == "v128Hz":
			self.data_length = 7680
			self.segment_length = {
				'delta': 240,
				'theta': 240,
				'alpha': 480,
				'beta': 960,
				'gamma': 1920,
				'high': 3840,
			}
		elif self.suffix == "v256Hz":
			self.data_length = 15360
			self.segment_length = None # TODO
		else:
			raise ValueError("Suffix must be v128Hz or v256Hz")

		# Prevents the FutureWarning: from loading without setting weights_only to True
		warnings.filterwarnings("ignore", category=FutureWarning)
		self._index_data()

	def _index_data(self):
		self.graphs = dict()
		self.epochs = []
		subjects = [f.path.split("/")[-1] for f in os.scandir(self.root) if f.is_dir()]
		subjects = subjects[:int(len(subjects) * self.frac)]
		self.subjects_idx = {subject: idx for idx, subject in enumerate(subjects)}
		print(f"Loading {len(subjects)} subjects")
		self.num_subjects = len(self.subjects_idx)

		with ThreadPoolExecutor() as executor:
			futures = [executor.submit(self._process_subject, subject) for subject in subjects]
			for future in tqdm(futures):
				future.result()
		self.length = len(self.epochs)

	def _process_subject(self, subject):
		subject_graph_folder = os.path.join(self.root, subject, f"graphs_{self.suffix}")
		wavelet_path = os.path.join(self.root, subject, f"wavelet_decompositions_{self.suffix}")
		if os.path.exists(subject_graph_folder) and len(os.listdir(subject_graph_folder)) > 0 and os.path.exists(wavelet_path) and len(os.listdir(wavelet_path)) > 0:
			subject_graph_path = os.listdir(subject_graph_folder)[0]
			self.graphs[subject_graph_path.split("_")[0]] = torch.load(os.path.join(subject_graph_folder, subject_graph_path), weights_only=False)
			wavelet_files = os.listdir(wavelet_path)
			subject_epochs = dict()
			for file_name in wavelet_files:
				attributes = file_name.split("_")
				epoch_idx = int(attributes[-1][:-3])
				band = attributes[-4]
				if band == "freq":
					band = "high"
				if epoch_idx not in subject_epochs:
					subject_epochs[epoch_idx] = dict()
					subject_epochs[epoch_idx]['graph_name'] = attributes[0]
				subject_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_path, file_name), weights_only=False)
			for epoch_idx, epoch_wavelet_dict in subject_epochs.items():
				if self.include_high:
					epoch_tuple = (epoch_wavelet_dict['graph_name'], subject, epoch_idx,
								   epoch_wavelet_dict['delta'][:, 3:3 + self.segment_length['delta']],  # 246
									epoch_wavelet_dict['theta'][:, 3:3 + self.segment_length['theta']], # 246
									epoch_wavelet_dict['alpha'][:, 3:3 + self.segment_length['alpha']], # 486
									epoch_wavelet_dict['beta'][:, 3:3 + self.segment_length['beta']],   # 926
									epoch_wavelet_dict['gamma'][:, 2:2 + self.segment_length['gamma']], # 1925
									epoch_wavelet_dict['high'][:, 2:2 + self.segment_length['high']]) # 3843
				else:
					epoch_tuple = (epoch_wavelet_dict['graph_name'], subject, epoch_idx,
								    epoch_wavelet_dict['delta'][:, 3:3 + self.segment_length['delta']],  # 246
									epoch_wavelet_dict['theta'][:, 3:3 + self.segment_length['theta']], # 246
									epoch_wavelet_dict['alpha'][:, 3:3 + self.segment_length['alpha']], # 486
									epoch_wavelet_dict['beta'][:, 3:3 + self.segment_length['beta']],   # 926
									epoch_wavelet_dict['gamma'][:, 2:2 + self.segment_length['gamma']]) # 1925
				self.epochs.append(epoch_tuple)

	def len(self):
		return self.length
	def get(self, idx):
		epoch_tuple = self.epochs[idx]
		graph = self.graphs[epoch_tuple[0]]

		if self.include_high:
			data = {
				"graph": graph,
				"subject_name": epoch_tuple[1],
				"subject_idx": self.subjects_idx[epoch_tuple[1]],
				"delta": epoch_tuple[3],
				"theta": epoch_tuple[4],
				"alpha": epoch_tuple[5],
				"beta": epoch_tuple[6],
				"gamma": epoch_tuple[7],
				"high": epoch_tuple[8],
			}
		else:
			data = {
				"graph": graph,
				"subject_name": epoch_tuple[1],
				"subject_idx": self.subjects_idx[epoch_tuple[1]],
				"delta": epoch_tuple[3],
				"theta": epoch_tuple[4],
				"alpha": epoch_tuple[5],
				"beta": epoch_tuple[6],
				"gamma": epoch_tuple[7],
			}
		return data
"""	
import torch
import os
import warnings
from torch.utils.data import ConcatDataset
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class WaveletPretrainDataset(Dataset):
    def __init__(self, root, include_high=False, frac=1.0, suffix="v128Hz", transform=None):
        super().__init__(root, transform)
        self.frac = frac
        self.suffix = suffix
        self.include_high = include_high
        self.root = root
        
        if self.suffix == "v128Hz":
            self.data_length = 7680
            self.segment_length = {
                'delta': 240,
                'theta': 240,
                'alpha': 480,
                'beta': 960,
                'gamma': 1920,
                'high': 3840,
            }
        elif self.suffix == "v256Hz":
            self.data_length = 15360
            self.segment_length = None # TODO
        else:
            raise ValueError("Suffix must be v128Hz or v256Hz")

        # Prevents the FutureWarning: from loading without setting weights_only to True
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

    def _index_data(self):
        """Index available data files without loading them into memory"""
        self.graph_paths = dict()
        self.epochs = []
        subjects = [f.path.split("/")[-1] for f in os.scandir(self.root) if f.is_dir()]
        subjects = subjects[:int(len(subjects) * self.frac)]
        self.subjects_idx = {subject: idx for idx, subject in enumerate(subjects)}
        print(f"Indexing {len(subjects)} subjects")
        self.num_subjects = len(self.subjects_idx)

        for subject in tqdm(subjects):
            self._index_subject(subject)
        
        self.length = len(self.epochs)

    def _index_subject(self, subject):
        """Index files for a subject without loading the actual data"""
        subject_graph_folder = os.path.join(self.root, subject, f"graphs_{self.suffix}")
        wavelet_path = os.path.join(self.root, subject, f"wavelet_decompositions_{self.suffix}")
        
        if os.path.exists(subject_graph_folder) and len(os.listdir(subject_graph_folder)) > 0 and os.path.exists(wavelet_path) and len(os.listdir(wavelet_path)) > 0:
            subject_graph_path = os.listdir(subject_graph_folder)[0]
            graph_name = subject_graph_path.split("_")[0]
            
            # Store path to graph file instead of loading it
            self.graph_paths[graph_name] = os.path.join(subject_graph_folder, subject_graph_path)
            
            wavelet_files = os.listdir(wavelet_path)
            subject_epochs = dict()
            
            for file_name in wavelet_files:
                attributes = file_name.split("_")
                epoch_idx = int(attributes[-1][:-3])
                band = attributes[-4]
                if band == "freq":
                    band = "high"
                if epoch_idx not in subject_epochs:
                    subject_epochs[epoch_idx] = dict()
                    subject_epochs[epoch_idx]['graph_name'] = attributes[0]
                
                # Store file path instead of loading data
                subject_epochs[epoch_idx][band] = os.path.join(wavelet_path, file_name)
            
            # Create epoch tuples with file paths
            for epoch_idx, epoch_files_dict in subject_epochs.items():
                if self.include_high:
                    epoch_tuple = (epoch_files_dict['graph_name'], subject, epoch_idx,
                                   epoch_files_dict.get('delta'),
                                   epoch_files_dict.get('theta'),
                                   epoch_files_dict.get('alpha'),
                                   epoch_files_dict.get('beta'),
                                   epoch_files_dict.get('gamma'),
                                   epoch_files_dict.get('high'))
                else:
                    epoch_tuple = (epoch_files_dict['graph_name'], subject, epoch_idx,
                                   epoch_files_dict.get('delta'),
                                   epoch_files_dict.get('theta'),
                                   epoch_files_dict.get('alpha'),
                                   epoch_files_dict.get('beta'),
                                   epoch_files_dict.get('gamma'))
                self.epochs.append(epoch_tuple)

    def _load_wavelet_data(self, file_path, band):
        """Load and process wavelet data from file"""
        if file_path is None:
            return None
        
        data = torch.load(file_path, weights_only=False)
        
        # Apply the same slicing as before
        if band in ['delta', 'theta', 'alpha', 'beta']:
            return data[:, 3:3 + self.segment_length[band]]
        elif band in ['gamma', 'high']:
            return data[:, 2:2 + self.segment_length[band]]
        else:
            return data

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph_name = epoch_tuple[0]
        
        # Load graph lazily
        if graph_name not in self.graph_paths:
            raise ValueError(f"Graph {graph_name} not found")
        graph = torch.load(self.graph_paths[graph_name], weights_only=False)

        # Load wavelet data lazily
        if self.include_high:
            data = {
                "graph": graph,
                "subject_name": epoch_tuple[1],
                "subject_idx": self.subjects_idx[epoch_tuple[1]],
                "delta": self._load_wavelet_data(epoch_tuple[3], 'delta'),
                "theta": self._load_wavelet_data(epoch_tuple[4], 'theta'),
                "alpha": self._load_wavelet_data(epoch_tuple[5], 'alpha'),
                "beta": self._load_wavelet_data(epoch_tuple[6], 'beta'),
                "gamma": self._load_wavelet_data(epoch_tuple[7], 'gamma'),
                "high": self._load_wavelet_data(epoch_tuple[8], 'high'),
            }
        else:
            data = {
                "graph": graph,
                "subject_name": epoch_tuple[1],
                "subject_idx": self.subjects_idx[epoch_tuple[1]],
                "delta": self._load_wavelet_data(epoch_tuple[3], 'delta'),
                "theta": self._load_wavelet_data(epoch_tuple[4], 'theta'),
                "alpha": self._load_wavelet_data(epoch_tuple[5], 'alpha'),
                "beta": self._load_wavelet_data(epoch_tuple[6], 'beta'),
                "gamma": self._load_wavelet_data(epoch_tuple[7], 'gamma'),
            }
        return data

class WaveletPretrainConcatDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        global_subject_idx = len(datasets[0].subjects_idx)
        modified_datasets = [datasets[0]]
        for dataset_idx in range(1, len(datasets)):
            datasets.subjects_idx = {subject: idx + global_subject_idx for subject, idx in datasets[dataset_idx].items()}
            modified_datasets.append(datasets[dataset_idx])
            global_subject_idx += len(datasets[dataset_idx].subjects_idx)
        self.datasets = ConcatDataset(modified_datasets)
    def len(self):
        return len(self.datasets)

    def get(self, idx):
        return self.datasets[idx]


if __name__ == "__main__":
	dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/TUH-128Hz", frac=0.001, include_high=True)

	print("Length of dataset: ", len(dataset))

	data = dataset[0]

	for key, value in data.items():
		if not isinstance(value, torch.Tensor):
			print(key, value)
		else:
			print(key, value.shape)

	print("Data: ", len(data),
		data['graph'],
		"Data Label:",
		data['graph'].y)

	hbn_dataset = WaveletPretrainDataset(root="/storage/ice1/shared/bmed6780/mip_group_6/ef/HBN-128Hz", frac=1.0, include_high=True)

	print("Length of dataset: ", len(hbn_dataset))

	data = hbn_dataset[0]

	for key, value in data.items():
		if not isinstance(value, torch.Tensor):
			print(key, value)
		else:
			print(key, value.shape)

	print("Data: ", len(data),
		data['graph'],
		"Data Label:",
		data['graph'].y)
