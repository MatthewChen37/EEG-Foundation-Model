import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_freq']

class WaveletMOBIDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None):
        super(WaveletMOBIDataset, self).__init__(root, transform)
        self.frac = frac
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
        for graph_file in os.listdir(graph_folder):
            if not graph_file.endswith("_graph.pt"):
                continue
            graph_name = "_".join(graph_file.split("_")[:2])
            graph_path = os.path.join(graph_folder, graph_file)
            graph = torch.load(graph_path)
            graph.y = torch.tensor([-1])
            self.graphs[graph_name] = graph

        for file_name in os.listdir(wavelet_folder):
            if not file_name.endswith(".pt"):
                continue
            parts = file_name.split("_")
            if len(parts) < 4:
                continue
            graph_name = "_".join(parts[:2])
            band = parts[-2]
            epoch_idx = int(parts[-3]) if parts[-3].isdigit() else 0
            if epoch_idx not in folder_epochs:
                folder_epochs[epoch_idx] = dict()
                folder_epochs[epoch_idx]['graph_name'] = graph_name
            folder_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_folder, file_name))

        for epoch_idx, band_dict in folder_epochs.items():
            if not all(b in band_dict for b in BANDS):
                continue
            self.epochs.append((band_dict['graph_name'], wavelet_folder, epoch_idx, *[band_dict[b] for b in BANDS]))

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph = self.graphs[epoch_tuple[0]]
        data = {
            "graph": graph,
            "wavelet_folder": epoch_tuple[1],
        }
        for i, band in enumerate(BANDS):
            data[band] = epoch_tuple[3 + i]
        return data