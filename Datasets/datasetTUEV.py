import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class WaveletTUEVDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None):
        super(WaveletTUEVDataset, self).__init__(root, transform)
        self.frac = frac
        # Prevents the FutureWarning: from loading without setting weights_only to True
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
        folder_graph_path = os.listdir(graph_folder)[0]
        split_path = folder_graph_path.split("_")
        folder_class = curr_folder.split("_")[-1]
        subject_int_class = 0 if folder_class == "normal" else 1
        graph = torch.load(os.path.join(graph_folder, folder_graph_path))
        graph.y = torch.tensor([subject_int_class])
        graph_name = "_".join(split_path[0:2])
        self.graphs[graph_name] = graph
        wavelet_files = os.listdir(wavelet_folder)
        for file_name in wavelet_files:
            attributes = file_name.split("_")
            epoch_idx = int(attributes[-1][:-3])
            band = attributes[-4]
            if epoch_idx not in folder_epochs:
                folder_epochs[epoch_idx] = dict()
                folder_epochs[epoch_idx]['graph_name'] = graph_name
            folder_epochs[epoch_idx][band] = torch.load(os.path.join(wavelet_folder, file_name))

    def len(self):
        return self.length