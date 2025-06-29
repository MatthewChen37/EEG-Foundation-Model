import torch
import torch.nn as nn
from Model.MENDR.mAtt.spd import SPDTangentSpace

class TUABFinetuneDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tangent = SPDTangentSpace(19)
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(nn.Linear(5*19*10, 1*19*10), nn.LayerNorm(1*19*10), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*10, 1*19*10))
        self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(1*19*10, 1))

        '''
        self.decoders = nn.ParameterDict({
            'delta': nn.Sequential(nn.Linear(5*19*32, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1), nn.LayerNorm(1*19*1)), 
            'theta': nn.Sequential(nn.Linear(5*19*32, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1), nn.LayerNorm(1*19*1)),
            'alpha': nn.Sequential(nn.Linear(5*19*64, 1*19*64), nn.LayerNorm(1*19*64), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*64, 1*19*1), nn.LayerNorm(1*19*1)),
            'beta': nn.Sequential(nn.Linear(5*19*128, 1*19*128), nn.LayerNorm(1*19*128), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*128, 1*19*1), nn.LayerNorm(1*19*1)),
            'gamma': nn.Sequential(nn.Linear(5*19*256, 1*19*256), nn.LayerNorm(1*19*256), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*256, 1*19*1), nn.LayerNorm(1*19*1)),
            #'high': nn.Sequential(nn.Linear(5*19*512, 1*19*512), nn.LayerNorm(1*19*512), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*512, 1*19*1), nn.LayerNorm(1*19*1)),
        })
        self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(5*19*1, 1))
        '''

    '''
    def forward(self, encodings):
        feautrizations = dict()
        for band, decoder in self.decoders.items():
            feautrizations[band] = decoder(self.flatten(encodings[band]))
        x = torch.cat(list(feautrizations.values()), dim=-1)
        return self.final_decoder(x)
    '''
        
    def forward(self, x):
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        embedding_dim = x.shape[2]
        x = self.tangent(x.view(batch_size*num_patches, embedding_dim, embedding_dim))
        x = x.view(batch_size, num_patches, -1)
        x = self.flatten(x)
        x = self.seq(x)
        return self.final_decoder(x)