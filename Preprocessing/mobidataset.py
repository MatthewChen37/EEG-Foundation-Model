import os
import torch
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high']

class WaveletMOBIDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None, split="train"):
        super(WaveletMOBIDataset, self).__init__(root, transform)
        self.frac = frac
        self.split = split
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

    def _index_data(self):
        self.graphs = dict()
        self.epochs = []
        folders = os.listdir(self.root)
        folders = folders[:int(len(folders) * self.frac)]
        print(f"Loading {len(folders)} folders...")

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [
                executor.submit(
                    self._process_folder,
                    folder,
                    os.path.join(self.root, folder, "wavelet_decompositions"),
                    os.path.join(self.root, folder, "graphs")
                )
                for folder in folders
            ]
            for future in tqdm(futures):
                future.result()
        self.length = len(self.epochs)

    def _process_folder(self, subject, wavelet_folder, graph_folder):
        if not os.path.exists(wavelet_folder) or not os.path.exists(graph_folder):
            return

        graph_files = os.listdir(graph_folder)
        for graph_file in graph_files:
            graph_name = "_".join(graph_file.split("_")[:-1])
            graph_path = os.path.join(graph_folder, graph_file)
            graph = torch.load(graph_path).clone()
            graph.x = None
            self.graphs[graph_name] = graph

        wavelet_files = os.listdir(wavelet_folder)
        epoch_dict = dict()
        for wf in wavelet_files:
            parts = wf.split("_")
            band = parts[-3]
            epoch_key = "_".join(parts[:-3])
            path = os.path.join(wavelet_folder, wf)
            epoch_dict.setdefault(epoch_key, {})[band] = torch.load(path)

        for epoch_key, band_data in epoch_dict.items():
            if all(b in band_data for b in BANDS) and epoch_key in self.graphs:
                self.epochs.append((
                    epoch_key,
                    wavelet_folder,
                    band_data['delta'],
                    band_data['theta'],
                    band_data['alpha'],
                    band_data['beta'],
                    band_data['gamma'],
                    band_data['high']
                ))

    def len(self):
        return self.length

    def get(self, idx):
        epoch_tuple = self.epochs[idx]
        graph_name, wavelet_folder, delta, theta, alpha, beta, gamma, high = epoch_tuple
        graph = self.graphs[graph_name].clone()
        return {
            "graph": graph,
            "wavelet_folder": wavelet_folder,
            "delta": delta,
            "theta": theta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "high": high,
        }
