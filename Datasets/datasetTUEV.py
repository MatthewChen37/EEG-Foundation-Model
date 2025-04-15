import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletTUEVDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None, split="train"):
        super(WaveletTUEVDataset, self).__init__(root, transform)
        self.frac = frac
        self.split = split
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
            futures = [executor.submit(self._process_folder, curr_folder, os.path.join(self.root, curr_folder, "wavelet_decompositions"), os.path.join(self.root, curr_folder, "graphs")) for curr_folder in folders]
            for future in tqdm(futures):
                future.result()
        self.length = len(self.epochs)

    def _process_folder(self, curr_folder, wavelet_folder, graph_folder):
        folder_bands = dict()
        folder_graph_path = os.listdir(graph_folder)[0]
        split_path = folder_graph_path.split("_")
        graph = torch.load(os.path.join(graph_folder, folder_graph_path))
        graph.x = None
        if self.split == "train":
            graph_name = "_".join(split_path[0:5])
        else:
            graph_name = "_".join(split_path[0:6])
        self.graphs[graph_name] = graph
        wavelet_files = os.listdir(wavelet_folder)
        for file_name in wavelet_files:
            attributes = file_name.split("_")
            band = attributes[-3]
            if band != "high":
                folder_bands[band] = torch.load(os.path.join(wavelet_folder, file_name))
        epoch_tuple = (
            graph_name,
            wavelet_folder,
            folder_bands['delta'],
            folder_bands['theta'],
            folder_bands['alpha'],
            folder_bands['beta'],
            folder_bands['gamma'],
        )
        self.epochs.append(epoch_tuple)

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph_name, wavelet_folder, delta, theta, alpha, beta, gamma = epoch_tuple
        graph = self.graphs[graph_name].clone()
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

if __name__ == "__main__":
    train_dataset = WaveletTUEVDataset(root="/home/hice1/mchen439/scratch/TUEV/train", frac=1.0, split="train")
    eval_dataset = WaveletTUEVDataset(root="/home/hice1/mchen439/scratch/TUEV/eval", frac=1.0, split="eval")

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

    print("Train Graphs: ", len(train_dataset.graphs)) # Should be both 83932
    print("Train Epochs: ", len(train_dataset.epochs)) # Should be both 83932

    print("Val Graphs: ", len(eval_dataset.graphs)) # Should be both 29421
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
