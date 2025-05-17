import torch
import torch.nn as nn
from mAtt.spd import SPDTangentSpace


class TUABFinetuneDecoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tangent = SPDTangentSpace(19, device=device)
        self.flatten = nn.Flatten().to(device)

        self.seq = nn.Sequential(nn.Linear(19 * 10, 19*10), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(19*10, 19*5), nn.GELU(), nn.Dropout(p=0.1))
        self.final_decoder = nn.Linear(19*5, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        embedding_dim = x.shape[2]
        x = self.tangent(x.view(batch_size*num_patches, embedding_dim, embedding_dim))
        x = self.flatten(x)
        x = self.seq(x)
        return self.final_decoder(x)