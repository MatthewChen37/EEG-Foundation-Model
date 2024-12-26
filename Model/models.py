
from copy import deepcopy

import numpy as np
import torch

import torch
import numpy as np
from torch import nn
from dataset import EEGDataset

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/models.py
2. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/layers.py
'''
class BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
    and *classification*.
    """
    def __init__(self, samples, channels, return_features=True):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):

        return None

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        print("Creating {} using: {} channels with trials of {} samples at {}Hz".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq))
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)


class Classifier(BaseModel):
    """
    A generic Classifer container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: EEGDataset, **modelargs):
        """
        Create a classifier from a dataset.

        Parameters
        ----------
        dataset
        modelargs: dict
                   Options to construct the dataset, if dataset does not have listed targets, targets must be specified
                   in the keyword arguments or will fall back to 2.

        Returns
        -------
        model: Classifier
               A new `Classifier` ready to classifiy data from `dataset`
        """
        if hasattr(dataset, 'get_targets'):
            targets = len(np.unique(dataset.get_targets()))
        elif dataset.info is not None and isinstance(dataset.info.targets, int):
            targets = dataset.info.targets
        else:
            targets = 2
        modelargs.setdefault('targets', targets)
        print("Creating {} using: {} channels x {} samples at {}Hz | {} targets".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq,
                                                                                        modelargs['targets']))
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)

    def __init__(self, targets, samples, channels, return_features=True):
        super(Classifier, self).__init__(samples, channels, return_features=return_features)
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

    def make_new_classification_layer(self):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.

        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.

        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the
        target should override this method, and there should be a variable called `self.classifier`

        """
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        super(Classifier, self).freeze_features(unfreeze=unfreeze)

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def classifier_forward(self, features):
        return self.classifier(features)

    def features_forward(self, x):
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)

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
        self.dropout = nn.Dropout2d(p=do_rate)
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
            nn.Dropout2d(do)
        )

    def forward(self, x):
        return torch.cat((x, self.net(x)), dim=1)


class DenseSpatialFilter(nn.Module):
    def __init__(self, channels, growth, depth, in_ch=1, bottleneck=4, dropout_rate=0.0, activation=nn.LeakyReLU,
                 collapse=True):
        """
        This extends the :any:`DenseFilter` to specifically operate in channel space and collapse this dimension
        over the course of `depth` layers.

        Parameters
        ----------
        channels
        growth
        depth
        in_ch
        bottleneck
        dropout_rate
        activation
        collapse
        """
        super().__init__()
        self.net = nn.Sequential(*[
            DenseFilter(in_ch + growth * d, growth, bottleneck=bottleneck, do=dropout_rate,
                        activation=activation) for d in range(depth)
        ])
        n_filters = in_ch + growth * depth
        self.collapse = collapse
        if collapse:
            self.channel_collapse = ConvBlock2D(n_filters, n_filters, (channels, 1), do_rate=0)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.net(x)
        if self.collapse:
            return self.channel_collapse(x).squeeze(-2)
        return x


class SpatialFilter(nn.Module):
    def __init__(self, channels, filters, depth, in_ch=1, dropout_rate=0.0, activation=nn.LeakyReLU, batch_norm=True,
                 residual=False):
        super().__init__()
        kernels = [(channels // depth, 1) for _ in range(depth-1)]
        kernels += [(channels - sum(x[0] for x in kernels) + depth-1, 1)]
        self.filter = nn.Sequential(
            ConvBlock2D(in_ch, filters, kernels[0], do_rate=dropout_rate/depth, activation=activation,
                        batch_norm=batch_norm),
            *[ConvBlock2D(filters, filters, kernel, do_rate=dropout_rate/depth, activation=activation,
                          batch_norm=batch_norm)
              for kernel in kernels[1:]]
        )
        self.residual = nn.Conv1d(channels * in_ch, filters, 1) if residual else None

    def forward(self, x):
        res = x
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        elif self.residual:
            res = res.contiguous().view(res.shape[0], -1, res.shape[3])
        x = self.filter(x).squeeze(-2)
        return x + self.residual(res) if self.residual else x


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