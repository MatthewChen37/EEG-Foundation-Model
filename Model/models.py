from abc import ABCMeta

import math

from copy import deepcopy

import numpy as np

from ..data.dataset import DN3ataset
from .layers import *

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/models.py
'''
class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`DN3BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
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
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)


class Classifier(DN3BaseModel):
    """
    A generic Classifer container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
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
        assert isinstance(dataset, DN3ataset)
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


class StrideClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, targets, samples, channels, stride_width=2, return_features=False):
        """
        Instead of summarizing the entire temporal dimension into a single prediction, a prediction kernel is swept over
        the final sequence representation and generates predictions at each step.

        Parameters
        ----------
        targets
        samples
        channels
        stride_width
        return_features
        """
        self.stride_width = stride_width
        super(StrideClassifier, self).__init__(targets, samples, channels, return_features=return_features)

    def make_new_classification_layer(self):
        self.classifier = torch.nn.Conv1d(self.num_features_for_classification, self.targets,
                                          kernel_size=self.stride_width)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


class LogRegNetwork(Classifier):
    """
    In effect, simply an implementation of linear kernel (multi)logistic regression
    """
    def features_forward(self, x):
        return x

    @property
    def num_features_for_classification(self):
        return self.samples * self.channels


class TIDNet(Classifier):
    """
    The Thinker Invariant Densenet from Kostas & Rudzicz 2020, https://doi.org/10.1088/1741-2552/abb7a7

    This alone is not strictly "thinker invariant", but on average outperforms shallower models at inter-subject
    prediction capability.
    """

    def __init__(self, targets, samples, channels, s_growth=24, t_filters=32, do=0.4, pooling=20,
                 activation=nn.LeakyReLU, temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3,
                 summary=-1, return_features=False):
        self.temp_len = math.ceil(temp_span * samples)
        summary = samples // pooling if summary == -1 else summary
        self._num_features = (t_filters + s_growth * spat_layers) * summary
        super().__init__(targets, samples, channels, return_features=return_features)

        self.temporal = nn.Sequential(
            Expand(axis=1),
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len, activation=activation),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(do),
        )

        self.spatial = DenseSpatialFilter(self.channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck, activation=activation)
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)),
        )

    @property
    def num_features_for_classification(self):
        return self._num_features

    def features_forward(self, x, **kwargs):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)

