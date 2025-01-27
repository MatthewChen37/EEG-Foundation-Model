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


class IndexSelect(nn.Module):
    def __init__(self, indices):
        super().__init__()
        assert isinstance(indices, (int, list, tuple))
        if isinstance(indices, int):
            indices = [indices]
        self.indices = list()
        for i in indices:
            assert isinstance(i, int)
            self.indices.append(i)

    def forward(self, *x):
        if len(x) == 1 and isinstance(x[0], tuple):
            x = x[0]
        if len(self.indices) == 1:
            return x[self.indices[0]]
        return [x[i] for i in self.indices]


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class ConvBlock2D(nn.Module):
    """
    Implements complete convolution block with order:
      - Convolution
      - dropout (spatial)
      - activation
      - batch-norm
      - (optional) residual reconnection
    """

    def __init__(self, in_filters, out_filters, kernel, stride=(1, 1), padding=0, dilation=1, groups=1, do_rate=0.5,
                 batch_norm=True, activation=nn.LeakyReLU, residual=False):
        super().__init__()
        self.kernel = kernel
        self.activation = activation()
        self.residual = residual

        self.conv = nn.Conv2d(in_filters, out_filters, kernel, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=not batch_norm)
        self.dropout = nn.Dropout1d(p=do_rate)
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, input, **kwargs):
        res = input
        input = self.conv(input, **kwargs)
        input = self.dropout(input)
        input = self.activation(input)
        input = self.batch_norm(input)
        return input + res if self.residual else input

class DenseFilter(nn.Module):
    def __init__(self, in_features, growth_rate, filter_len=5, do=0.5, bottleneck=2, activation=nn.LeakyReLU, dim=-2):
        """
        This DenseNet-inspired filter block features in the TIDNet network from Kostas & Rudzicz 2020 (Thinker
        Invariance). 2D convolution is used, but with a kernel that only spans one of the dimensions. In TIDNet it is
        used to develop channel operations independently of temporal changes.

        Parameters
        ----------
        in_features
        growth_rate
        filter_len
        do
        bottleneck
        activation
        dim
        """
        super().__init__()
        dim = dim if dim > 0 else dim + 4
        if dim < 2 or dim > 3:
            raise ValueError('Only last two dimensions supported')
        kernel = (filter_len, 1) if dim == 2 else (1, filter_len)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_features),
            activation(),
            nn.Conv2d(in_features, bottleneck * growth_rate, 1),
            nn.BatchNorm2d(bottleneck * growth_rate),
            activation(),
            nn.Conv2d(bottleneck * growth_rate, growth_rate, kernel, padding=tuple((k // 2 for k in kernel))),
            nn.Dropout1d(do)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)



class TemporalFilter(nn.Module):

    def __init__(self, channels, filters, depth, temp_len, dropout=0., activation=nn.LeakyReLU, residual='netwise'):
        """
        This implements the dilated temporal-only spanning convolution from TIDNet.

        Parameters
        ----------
        channels
        filters
        depth
        temp_len
        dropout
        activation
        residual
        """
        super().__init__()
        temp_len = temp_len + 1 - temp_len % 2
        self.residual_style = str(residual)
        net = list()

        for i in range(depth):
            dil = depth - i
            conv = nn.utils.weight_norm(nn.Conv2d(channels if i == 0 else filters, filters, kernel_size=(1, temp_len),
                                      dilation=dil, padding=(0, dil * (temp_len - 1) // 2)))
            net.append(nn.Sequential(
                conv,
                activation(),
                nn.Dropout2d(dropout)
            ))
        if self.residual_style.lower() == 'netwise':
            self.net = nn.Sequential(*net)
            self.residual = nn.Conv2d(channels, filters, (1, 1))
        elif residual.lower() == 'dense':
            self.net = net

    def forward(self, x):
        if self.residual_style.lower() == 'netwise':
            return self.net(x) + self.residual(x)
        elif self.residual_style.lower() == 'dense':
            for l in self.net:
                x = torch.cat((x, l(x)), dim=1)
            return x