import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'graph_name']

class WaveletMOBIDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None):
        super(WaveletMOBIDataset, self).__init__(root, transform)
        self.frac = frac
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

    def _index_data(self):
        self.graphs = dict()
        self.gaits = dict()
        self.epochs = []
        folders = os.listdir(self.root)
        folders = folders[:int(len(folders) * self.frac)]
        print(f"Loading {len(folders)} folders")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self._process_folder, curr_folder, os.path.join(self.root, curr_folder, "wavelet_decompositions"), os.path.join(self.root, curr_folder, "graphs"), os.path.join(self.root, curr_folder, "gait")) for curr_folder in folders]
            for future in tqdm(futures):
                future.result()
        self.length = len(self.epochs)

    def _process_folder(self, curr_folder, wavelet_folder, graph_folder, gait_folder):
        #print(f"Processing {curr_folder}", wavelet_folder, graph_folder)
        folder_epochs = dict()
        for graph_file in os.listdir(graph_folder):
            if not graph_file.endswith(".pt"):
                continue
            graph_name = "_".join(graph_file.split("_")[:2])
            graph_path = os.path.join(graph_folder, graph_file)
            graph = torch.load(graph_path, weights_only=False)
            graph.y = torch.tensor([-1])
            self.graphs[graph_name] = graph

        for gait_file in os.listdir(gait_folder):
            if not gait_file.endswith(".pt"):
                continue
            gait_name = "_".join(gait_file.split("_")[:2])
            gait_path = os.path.join(gait_folder, gait_file)
            gait = torch.load(gait_path)
            self.gaits[graph_name] = gait

        for file_name in os.listdir(wavelet_folder):
            if not file_name.endswith(".pt"):
                continue
            parts = file_name.split("_")
            if len(parts) < 4:
                continue
            graph_name = "_".join(parts[:2])
            band = parts[2]
            epoch_idx = int(parts[-1][:-3])
            if epoch_idx not in folder_epochs:
                folder_epochs[epoch_idx] = dict()
                folder_epochs[epoch_idx]['graph_name'] = graph_name
            folder_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_folder, file_name))

        for epoch_idx, band_dict in folder_epochs.items():
            #print(band_dict.keys())
            if not all(b in band_dict for b in BANDS):
                continue
            self.epochs.append((band_dict['graph_name'], wavelet_folder, epoch_idx, *[band_dict[b] for b in BANDS]))
            #print("HERE")
    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph = self.graphs[epoch_tuple[0]]
        graph.y = self.gaits[epoch_tuple[0]]
        data = {
            "graph": graph,
            "wavelet_folder": epoch_tuple[1],
        }
        for i, band in enumerate(BANDS):
            if band == "delta":
                data[band] = epoch_tuple[3 + i][:, 3:27]
            elif band == "theta":
                data[band] = epoch_tuple[3 + i][:, 3:27]
            elif band == "alpha":
                data[band] = epoch_tuple[3 + i][:, 3:51]
            elif band == "beta":
                data[band] = epoch_tuple[3 + i][:, 3:99]
            elif band == "gamma":
                data[band] = epoch_tuple[3 + i][:, 2:194]
            else: 
                data[band] = epoch_tuple[3 + i]
        return data

if __name__ == "__main__":
    train_dataset = WaveletMOBIDataset(root="/home/hice1/mchen439/scratch/MOBI/train/graphs")
    eval_dataset = WaveletMOBIDataset(root="/home/hice1/mchen439/scratch/MOBI/eval/graphs")

    print("Length of train dataset: ", len(train_dataset))
    print("Length of val dataset: ", len(eval_dataset))
    data = train_dataset[0]
    print("Data: ", len(data), data['graph'], data['wavelet_folder'],
                    "delta", data['delta'].shape,
                    "theta", data['theta'].shape,
                    "alpha", data['alpha'].shape,
                    "beta", data['beta'].shape,
                    "gamma", data['gamma'].shape,
                    "Data Label:", data['graph'].y.shape,
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