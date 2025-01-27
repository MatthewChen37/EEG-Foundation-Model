import torch
from torch import nn

'''
Useful layers for building models.

Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/models.py
2. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/layers.py
'''
class _SingleAxisOperation(nn.Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        raise NotImplementedError

# Some general purpose convenience layers
# ---------------------------------------

class Expand(_SingleAxisOperation):
    def forward(self, x):
        return x.unsqueeze(self.axis)


class Squeeze(_SingleAxisOperation):
    def forward(self, x):
        return x.squeeze(self.axis)

class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)

class Concatenate(_SingleAxisOperation):
    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        return torch.cat(x, dim=self.axis)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)
