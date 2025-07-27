import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletTUEVDataset(Dataset):
    def __init__(self, root, include_high=False, frac=1.0, transform=None, split="train"):
        super(WaveletTUEVDataset, self).__init__(root, transform)
        self.frac = frac
        self.split = split
        self.include_high = include_high
        # Prevents the FutureWarning: from loading without setting weights_only to True
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

    def _index_data(self):
        self.graph_paths = dict()
        self.epochs = []
        folders = os.listdir(self.root)
        folders = folders[:int(len(folders) * self.frac)]
        print(f"Indexing {len(folders)} folders")

        # Remove ThreadPoolExecutor since we're only indexing, not loading
        for curr_folder in tqdm(folders):
            wavelet_folder = os.path.join(self.root, curr_folder, "wavelet_decompositions")
            graph_folder = os.path.join(self.root, curr_folder, "graphs")
            self._index_folder(wavelet_folder, graph_folder)
        
        self.length = len(self.epochs)

    def _index_folder(self, wavelet_folder, graph_folder):
        folder_bands = dict()
        folder_graph_path = os.listdir(graph_folder)[0]
        split_path = folder_graph_path.split("_")
        
        # Store graph path instead of loading
        if self.split == "train":
            graph_name = "_".join(split_path[0:5])
        else:
            graph_name = "_".join(split_path[0:7])
        
        self.graph_paths[graph_name] = os.path.join(graph_folder, folder_graph_path)
        
        # Store file paths instead of loading wavelet data
        wavelet_files = os.listdir(wavelet_folder)
        for file_name in wavelet_files:
            attributes = file_name.split("_")
            band = attributes[-3]
            if band != "freq":
                folder_bands[band] = os.path.join(wavelet_folder, file_name)
            elif band == "freq" and self.include_high:
                folder_bands["high"] = os.path.join(wavelet_folder, file_name)

        # Create epoch tuple with file paths
        if self.include_high:
            epoch_tuple = (
                graph_name,
                wavelet_folder,
                folder_bands.get('delta'),
                folder_bands.get('theta'),
                folder_bands.get('alpha'),
                folder_bands.get('beta'),
                folder_bands.get('gamma'),
                folder_bands.get('high'),
            )
        else:
            epoch_tuple = (
                graph_name,
                wavelet_folder,
                folder_bands.get('delta'),
                folder_bands.get('theta'),
                folder_bands.get('alpha'),
                folder_bands.get('beta'),
                folder_bands.get('gamma'),
            )
        self.epochs.append(epoch_tuple)

    def _load_band_data(self, file_path, band):
        """Load and slice wavelet data for a specific band"""
        if file_path is None:
            return None
        
        data = torch.load(file_path, weights_only=False)
        
        # Apply the same slicing as in the original implementation
        if band == 'delta':
            return data[:, 3:27]
        elif band == 'theta':
            return data[:, 3:27]
        elif band == 'alpha':
            return data[:, 3:51]
        elif band == 'beta':
            return data[:, 3:99]
        elif band == 'gamma':
            return data[:, 2:194]
        elif band == 'high':
            return data[:, 2:386]
        else:
            return data

    def _load_graph(self, graph_name):
        """Load graph from file path"""
        if graph_name not in self.graph_paths:
            raise ValueError(f"Graph {graph_name} not found")
        
        graph = torch.load(self.graph_paths[graph_name], weights_only=False).clone()
        graph.x = None
        return graph

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph_name = epoch_tuple[0]
        wavelet_folder = epoch_tuple[1]
        
        # Load graph lazily
        graph = self._load_graph(graph_name)

        # Load wavelet data lazily
        if self.include_high:
            data = {
                "graph": graph,
                "wavelet_folder": wavelet_folder,
                "delta": self._load_band_data(epoch_tuple[2], 'delta'),
                "theta": self._load_band_data(epoch_tuple[3], 'theta'),
                "alpha": self._load_band_data(epoch_tuple[4], 'alpha'),
                "beta": self._load_band_data(epoch_tuple[5], 'beta'),
                "gamma": self._load_band_data(epoch_tuple[6], 'gamma'),
                "high": self._load_band_data(epoch_tuple[7], 'high'),
            }
        else:
            data = {
                "graph": graph,
                "wavelet_folder": wavelet_folder,
                "delta": self._load_band_data(epoch_tuple[2], 'delta'),
                "theta": self._load_band_data(epoch_tuple[3], 'theta'),
                "alpha": self._load_band_data(epoch_tuple[4], 'alpha'),
                "beta": self._load_band_data(epoch_tuple[5], 'beta'),
                "gamma": self._load_band_data(epoch_tuple[6], 'gamma'),
            }
        return data

'''
import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletTUEVDataset(Dataset):
    def __init__(self, root, include_high=False, frac=1.0, transform=None, split="train"):
        super(WaveletTUEVDataset, self).__init__(root, transform)
        self.frac = frac
        self.split = split
        self.include_high = include_high
        # Prevents the FutureWarning: from loading without setting weights_only to True
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

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
        folder_bands = dict()
        folder_graph_path = os.listdir(graph_folder)[0]
        split_path = folder_graph_path.split("_")
        graph = torch.load(os.path.join(graph_folder, folder_graph_path)).clone() # I hypothesize that this might be the cause of cloning for the MENDREncoder comment
        graph.x = None
        if self.split == "train":
            graph_name = "_".join(split_path[0:5])
        else:
            graph_name = "_".join(split_path[0:7])
        self.graphs[graph_name] = graph
        wavelet_files = os.listdir(wavelet_folder)
        for file_name in wavelet_files:
            attributes = file_name.split("_")
            band = attributes[-3]
            if band != "freq":
                folder_bands[band] = torch.load(os.path.join(wavelet_folder, file_name))
            elif band == "freq" and self.include_high:
                folder_bands["high"] = torch.load(os.path.join(wavelet_folder, file_name))

        if self.include_high:
            epoch_tuple = (
                graph_name,
                wavelet_folder,
                folder_bands['delta'][:, 3: 27],
                folder_bands['theta'][:, 3: 27],
                folder_bands['alpha'][:, 3: 51],
                folder_bands['beta'][:,  3: 99],
                folder_bands['gamma'][:, 3: 195],
                folder_bands['high'][:,  3: 387],
            )
        else:
            epoch_tuple = (
                graph_name,
                wavelet_folder,
                folder_bands['delta'][:, 3: 27],
                folder_bands['theta'][:, 3: 27],
                folder_bands['alpha'][:, 3: 51],
                folder_bands['beta'][:,  3: 99],
                folder_bands['gamma'][:, 3: 195],
                folder_bands['high'][:,  3: 387],
            )
        self.epochs.append(epoch_tuple)

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        if self.include_high:
            graph_name, wavelet_folder, delta, theta, alpha, beta, gamma, high = epoch_tuple
        else:
            graph_name, wavelet_folder, delta, theta, alpha, beta, gamma = epoch_tuple
        graph = self.graphs[graph_name].clone()

        if self.include_high:
            data = {
                "graph": graph,
                "wavelet_folder": wavelet_folder,
                "delta": delta,
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "high": high,
            }
        else:
            data = {
                "graph": graph,
                "wavelet_folder": wavelet_folder,
                "delta": delta,
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
        return data
'''


if __name__ == "__main__":
    train_dataset = WaveletTUEVDataset(root="/home/hice1/mchen439/scratch/TUEV_small/train", include_high=True, frac=1.0, split="train")
    eval_dataset = WaveletTUEVDataset(root="/home/hice1/mchen439/scratch/TUEV_small/eval", include_high=True, frac=1.0, split="eval")

    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(eval_dataset))
    data = train_dataset[0]
    print("Data: ", len(data), data['graph'], data['wavelet_folder'],
                    "delta", data['delta'].shape,
                    "theta", data['theta'].shape,
                    "alpha", data['alpha'].shape,
                    "beta", data['beta'].shape,
                    "gamma", data['gamma'].shape,
                    "high", data['high'].shape,
                    "Data Label:", data['graph'].y,
                    data['graph'].edge_index.shape,
                    data['graph'].edge_attr.shape)

    edge_index = data['graph'].edge_index
    edge_dist = data['graph'].edge_attr

    print("Train Graphs: ", len(train_dataset.graph_paths)) # Should be both 83932
    print("Train Epochs: ", len(train_dataset.epochs)) # Should be both 83932

    print("Val Graphs: ", len(eval_dataset.graph_paths)) # Should be both 29421
    print("Val Epochs: ", len(eval_dataset.epochs))

    from torch_geometric.loader import DataLoader
    finetune_train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)
    finetune_eval_loader = DataLoader(eval_dataset, batch_size=4, num_workers=1, shuffle=True, persistent_workers=True)
    print(len(finetune_train_loader), len(finetune_eval_loader))

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

    print("All tests passed")
