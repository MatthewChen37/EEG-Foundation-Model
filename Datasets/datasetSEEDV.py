import torch
import os
import warnings
from torch_geometric.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class WaveletSEEDVDataset(Dataset):
    def __init__(self, root, frac=1.0, transform=None):
        super(WaveletSEEDVDataset, self).__init__(root, transform)
        self.frac = frac
        warnings.filterwarnings("ignore", category=FutureWarning)
        self._index_data()

    def _index_data(self):
        self.graph_paths = dict()
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
        folder_graph_path = os.listdir(graph_folder)[0]
        split_path = folder_graph_path.split("_")
        graph = torch.load(os.path.join(graph_folder, folder_graph_path)).clone() 
        graph.x = None
        graph_name = "_".join(split_path[0:5])
        self.graph_paths[graph_name] = os.path.join(graph_folder, folder_graph_path)
        wavelet_files = os.listdir(wavelet_folder)
        for file_name in wavelet_files:
            attributes = file_name.split("_")
            band = attributes[-3]
            if band != "freq":
                folder_bands = torch.load(os.path.join(wavelet_folder, file_name))
            elif band == "freq":
                folder_bands = torch.load(os.path.join(wavelet_folder, file_name))


if __name__ == "__main__":
    train_dataset = WaveletSEEDVDataset(root="/home/hice1/mchen439/scratch/SEEDV_processed/train", frac=1.0)
    eval_dataset = WaveletSEEDVDataset(root="/home/hice1/mchen439/scratch/SEEDV_processed/eval", frac=1.0)

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