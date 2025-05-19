import torch
from torch.utils.data import ConcatDataset

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

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