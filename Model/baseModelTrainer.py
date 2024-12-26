import torch
import torch.nn as nn
import re
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sys import gettrace
import numpy as np
from transforms import BatchTransform
from pandas import DataFrame
from models import Classifier

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/processes.py
'''
class BaseModelTrainer(object):

    def __init__(self, lr=0.001, l2_weight_decay=0.01, cuda=None, **kwargs):
        """
        By default uses the SGD with momentum optimization.

        Parameters
        ----------
        cuda : bool, string, None
            If boolean, sets whether to enable training on the GPU, if a string, specifies can be used to specify
            which device to use. If None (default) figures it out automatically.
        lr : float
            The learning rate to use, this will probably something that should be tuned for each application.
            Start with multiplying or dividing by values of 2, 5 or 10 to seek out a good number.
        l2_weight_decay : float
            One of the simplest and most common regularizing techniques. If you find a model rapidly
            reaching high training accuracy (and not validation) increase this. If having trouble fitting
            the training data, decrease this.
        kwargs : dict
            Additional arguments.
        """
        if cuda is None:
            cuda = torch.cuda.is_available()
            if cuda:
                tqdm.write("GPU(s) detected: training and model execution will be performed on GPU.")
        if isinstance(cuda, bool):
            cuda = "cuda" if cuda else "cpu"
        assert isinstance(cuda, str)
        self.cuda = cuda
        self.device = torch.device(cuda)
        
        _before_members = set(self.__dict__.keys())
        self.__dict__.update(**kwargs)
        new_members = set(self.__dict__.keys()).difference(_before_members)
        self._training = False
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
                    self._trainables.append(member)
                self.__dict__[member] = self.__dict__[member].to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), weight_decay=l2_weight_decay, lr=lr, nesterov=True,
                                         momentum=0.9)
        self.scheduler = None
        self.scheduler_after_batch = False
        self.epoch = None
        self.lr = lr
        self.weight_decay = l2_weight_decay

        self._batch_transforms = list()
        self._eval_transforms = list()

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        del self.optimizer
        self.optimizer = optimizer
        self.lr = float(self.optimizer.param_groups[0]['lr'])

    def set_scheduler(self, scheduler, step_every_batch=False):
        """
        This allow the addition of a learning rate schedule to the process. By default, a linear warmup with cosine
        decay will be used. Any scheduler that is an instance of :any:`Scheduler` (pytorch's schedulers, or extensions
        thereof) can be set here. Additionally, a string keywords can be used including:
          - "constant"

        Parameters
        ----------
        scheduler: str, Scheduler
        step_every_batch: bool
                          Whether to call step after every batch (if `True`), or after every epoch (`False`)

        """
        if isinstance(scheduler, str):
            if scheduler.lower() == 'constant':
                scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: 1.0)
            else:
                raise ValueError("Scheduler {} is not supported.".format(scheduler))
        # This is the most common one that needs this, force this to be true
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler_after_batch = True
        else:
            self.scheduler_after_batch = step_every_batch
        self.scheduler = scheduler    

    def _optimize_dataloader_kwargs(self, num_worker_cap=6, **loader_kwargs):
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        # Use multiple worker processes when NOT DEBUGGING
        if gettrace() is None:
            try:
                # Find number of cpus available (taken from second answer):
                # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
                m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                              open('/proc/self/status').read())
                nw = bin(int(m.group(1).replace(',', ''), 16)).count('1')
                # Cap the number of workers at 6 (actually 4) to avoid pummeling disks too hard
                nw = min(num_worker_cap, nw)
            except FileNotFoundError:
                # Fallback for when proc/self/status does not exist
                nw = 2
        else:
            # 0 workers means not extra processes are spun up
            nw = 2
        loader_kwargs.setdefault('num_workers', int(nw - 2))
        print("Loading data with {} additional workers".format(loader_kwargs['num_workers']))
        return loader_kwargs

    def _get_batch(self, iterator):
        batch = [x.to(self.device, non_blocking=self.cuda == 'cuda') for x in next(iterator)]
        xforms = self._batch_transforms if self._training else self._eval_transforms
        for xform in xforms:
            if xform.only_trial_data:
                batch[0] = xform(batch[0])
            else:
                batch = xform(batch)
        return batch

    def add_batch_transform(self, transform: BatchTransform, training_only=True):
        self._batch_transforms.append(transform)
        if not training_only:
            self._eval_transforms.append(transform)

    def clear_batch_transforms(self):
        self._batch_transforms = list()
        self._eval_transforms = list()

    def parameters(self):
        """
        All the trainable parameters in the Trainable. This includes any architecture parameters and meta-parameters.

        Returns
        -------
        params :
                 An iterator of parameters
        """
        for member in self._trainables:
            yield from self.__dict__[member].parameters()

    def forward(self, *inputs):
        """
        Given a batch of inputs, return the outputs produced by the trainable module.

        Parameters
        ----------
        inputs :
               Tensors needed for underlying module.

        Returns
        -------
        outputs :
                Outputs of module

        """
        raise NotImplementedError

    def calculate_loss(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the loss.

        Returns
        -------
        Loss :
             Single loss quantity to be minimized.
        """
        if isinstance(outputs, (tuple, list)):
            device = outputs[0].device
        else:
            device = outputs.device
        loss_fn = self.loss
        if hasattr(self.loss, 'to'):
            loss_fn = loss_fn.to(device)
        return loss_fn(outputs, inputs[-1])

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)

    def train_step(self, *inputs):
        self.train(True)
        outputs = self.forward(*inputs)
        loss = self.calculate_loss(inputs, outputs)
        self.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_after_batch:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs)
        train_metrics.setdefault('loss', loss.item())

        return train_metrics

    def evaluate(self, dataset, **loader_kwargs):
        """
        Calculate and return metrics for a dataset

        Parameters
        ----------
        dataset: EEGDataset
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        metrics : OrderedDict
                Metric scores for the entire
        """
        self.train(False)
        inputs, outputs = self.predict(dataset, **loader_kwargs)
        metrics = self.calculate_metrics(inputs, outputs)
        metrics['loss'] = self.calculate_loss(inputs, outputs).item()
        return metrics

    def predict(self, dataset, **loader_kwargs):
        """
        Determine the outputs for all loaded data from the dataset

        Parameters
        ----------
        dataset: EEGDataset
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        inputs : Tensor
                 The exact inputs used to calculate the outputs (in case they were stochastic and need saving)
        outputs : Tensor
                  The outputs from each run of :function:`forward`
        """
        self.train(False)
        loader_kwargs.setdefault('batch_size', 1)
        dataset = self._make_dataloader(dataset, **loader_kwargs)

        pbar = tqdm.trange(len(dataset), desc="Predicting")
        data_iterator = iter(dataset)

        inputs = list()
        outputs = list()

        with torch.no_grad():
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                output_batch = self.forward(*input_batch)

                inputs.append([tensor.cpu() for tensor in input_batch])
                if isinstance(output_batch, torch.Tensor):
                    outputs.append(output_batch.cpu())
                else:
                    outputs.append([tensor.cpu() for tensor in output_batch])

        def package_multiple_tensors(batches: list):
            if isinstance(batches[0], torch.Tensor):
                return torch.cat(batches)
            elif isinstance(batches[0], (tuple, list)):
                return [torch.cat(b) for b in zip(*batches)]

        return package_multiple_tensors(inputs), package_multiple_tensors(outputs)

    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            elif m == 'lr':
                start_message += " {}: {:.3e} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.3f} |".format(m, metrics[m])
        tqdm.write(start_message)

    def save_best(self):
        """
        Create a snapshot of what is being currently trained for re-laoding with the :py:meth:`load_best()` method.

        Returns
        -------
        best : Any
               Whatever format is needed for :py:meth:`load_best()`, will be the argument provided to it.
        """
        return [{k: v.cpu() for k, v in self.__dict__[m].state_dict().items()} for m in self._trainables]

    def load_best(self, best):
        """
        Load the parameters as saved by :py:meth:`save_best()`.

        Parameters
        ----------
        best: Any
        """
        for m, state_dict in zip(self._trainables, best):
            self.__dict__[m].load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def _retain_best(self, old_checkpoint, metrics_to_check: dict, retain_string: str):
        if retain_string is None:
            return old_checkpoint
        best_checkpoint = old_checkpoint

        def found_best():
            tqdm.tqdm.write("Best {}. Retaining checkpoint...".format(retain_string))
            self.best_metric = metrics_to_check[retain_string]
            return self.save_best()

        if retain_string not in metrics_to_check.keys():
            tqdm.tqdm.write("No metric {} found in recorded metrics. Not saving best.")
        if self.best_metric is None:
            best_checkpoint = found_best()
        elif retain_string == 'loss' and metrics_to_check[retain_string] <= self.best_metric:
            best_checkpoint = found_best()
        elif retain_string != 'loss' and metrics_to_check[retain_string] >= self.best_metric:
            best_checkpoint = found_best()

        return best_checkpoint

    @staticmethod
    def _dataloader_args(dataset, training=False, **loader_kwargs):
        # Only shuffle and drop last when training
        loader_kwargs.setdefault('shuffle', training)
        loader_kwargs.setdefault('drop_last', training)

        return loader_kwargs

    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        """Any args that make more sense as a convenience function to be set"""
        if isinstance(dataset, DataLoader):
            return dataset

        return DataLoader(dataset, **self._dataloader_args(dataset, training, **loader_kwargs))
    
    def fit(self, training_dataset, epochs=1, validation_dataset=None,
            resume_epoch=None, resume_iteration=None, batch_size=8, warmup_frac=0.2,
            retain_best='loss', validation_interval=None, train_log_interval=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : EEGDataset
        validation_dataset : EEGDataset
        epochs : int
                 Total number of epochs to fit
        resume_epoch : int
                      The starting epoch to train from. This will likely only be used to resume training at a certain
                      point.
        resume_iteration : int
                          Similar to start epoch but specified in batches. This can either be used alone, or in
                          conjunction with `start_epoch`. If used alone, the start epoch is the floor of
                          `start_iteration` divided by batches per epoch. In other words this specifies cumulative
                          batches if start_epoch is not specified, and relative to the current epoch otherwise.
        batch_size : int
                     The batch_size to be used for the training and validation datasets.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        validation_interval: int, None
                             The number of batches between checking the validation dataset
        train_log_interval: int, None
                      The number of batches between persistent logging of training metrics, if None (default) happens
                      at the end of every epoch.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataset = self._make_dataloader(training_dataset, training=True, **loader_kwargs)

        if resume_epoch is None:
            if resume_iteration is None or resume_iteration < len(training_dataset):
                resume_epoch = 1
            else:
                resume_epoch = resume_iteration // len(training_dataset)
        resume_iteration = 1 if resume_iteration is None else resume_iteration % len(training_dataset)

        _clear_scheduler_after = self.scheduler is None
        if _clear_scheduler_after:
            last_epoch_workaround = len(training_dataset) * (resume_epoch - 1) + resume_iteration
            last_epoch_workaround = -1 if last_epoch_workaround <= 1 else last_epoch_workaround
            self.set_scheduler(
                torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, epochs=epochs,
                                                    steps_per_epoch=len(training_dataset),
                                                    pct_start=warmup_frac,
                                                    last_epoch=last_epoch_workaround)
            )

        validation_log = list()
        train_log = list()
        best_model = self.save_best()

        train_log_interval = len(training_dataset) if train_log_interval is None else train_log_interval

        def _validation(epoch, iteration=None):
            _metrics = self.evaluate(validation_dataset, **loader_kwargs)
            if iteration is not None:
                self.standard_logging(_metrics, "Validation: Epoch {} - Iteration {}".format(epoch, iteration))
            else:
                self.standard_logging(_metrics, "Validation: End of Epoch {}".format(epoch))
            _metrics['epoch'] = epoch
            validation_log.append(_metrics)
            return _metrics

        epoch_bar = tqdm.trange(resume_epoch, epochs + 1, desc="Epoch", unit='epoch', initial=resume_epoch, total=epochs)
        for epoch in epoch_bar:
            self.epoch = epoch
            pbar = tqdm.trange(resume_iteration, len(training_dataset) + 1, desc="Iteration", unit='batches',
                               initial=resume_iteration, total=len(training_dataset))
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                if isinstance(validation_interval, int) and (iteration % validation_interval == 0) and validation_dataset is not None:
                    _m = _validation(epoch, iteration)
                    best_model = self._retain_best(best_model, _m, retain_best)

            if validation_dataset is not None:
                metrics = _validation(epoch)
                best_model = self._retain_best(best_model, metrics, retain_best)

            # All future epochs should not start offset in iterations
            resume_iteration = 1

            if not self.scheduler_after_batch and self.scheduler is not None:
                tqdm.tqdm.write(f"Step {self.scheduler.get_last_lr()} {self.scheduler.last_epoch}")
                self.scheduler.step()

        if _clear_scheduler_after:
            self.set_scheduler(None)
        self.epoch = None

        if retain_best is not None and validation_dataset is not None:
            tqdm.tqdm.write("Loading best model...")
            self.load_best(best_model)

        return DataFrame(train_log), DataFrame(validation_log)

class StandardClassification(BaseModelTrainer):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=None, metrics=None, learning_rate=0.01,
                 label_smoothing=None, **kwargs):
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super(StandardClassification, self).__init__(cuda=cuda, lr=learning_rate, classifier=classifier,
                                                     metrics=metrics, **kwargs)
        if loss_fn is None:
            self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = loss_fn.to(self.device)
        self.best_metric = None

    @staticmethod
    def _simple_accuracy(inputs, outputs):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        # average over last dimensions
        while len(outputs.shape) >= 3:
            outputs = outputs.mean(dim=-1)
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def forward(self, *inputs):
        if isinstance(self.classifier, Classifier) and self.classifier.return_features:
            prediction, _ = self.classifier(*inputs[:-1])
        else:
            prediction = self.classifier(*inputs[:-1])
        return prediction

    def calculate_loss(self, inputs, outputs):
        inputs = list(inputs)

        def expand_for_strided_loss(factors):
            inputs[-1] = inputs[-1].unsqueeze(-1).expand(-1, *factors)

        check_me = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if len(check_me.shape) >= 3:
            expand_for_strided_loss(check_me.shape[2:])

        return super(StandardClassification, self).calculate_loss(inputs, outputs)

    def fit(self, training_dataset, epochs, validation_dataset=None, batch_size=8, warmup_frac=0.2,
             retain_best='loss', balance_method=None, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : EEGDataset
        validation_dataset : EEGDataset
        epochs : int
                Total number of epochs to fit
        batch_size : int
                     The batch_size to be used for the training and validation datasets.
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        balance_method : (None, str)
                         If and how to balance training samples when training. `None` (default) will simply randomly
                         sample all training samples equally. 'undersample' will sample each class N_min times
                         where N_min is equal to the number of examples in the minority class. 'oversample' will sample
                         each class N_max times, where N_max is the number of the majority class.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Notes
        -----
        Optimized data loading by setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. 

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        return super(StandardClassification, self).fit(training_dataset, epochs=epochs, batch_size=batch_size,
                                                       warmup_frac=warmup_frac, retain_best=retain_best,
                                                       validation_dataset=validation_dataset,
                                                       balance_method=balance_method,
                                                       **loader_kwargs)

    BALANCE_METHODS = ['undersample', 'oversample', 'ldam']
    def _make_dataloader(self, dataset, training=False, **loader_kwargs):
        if isinstance(dataset, DataLoader):
            return dataset

        loader_kwargs = self._dataloader_args(dataset, training=training, **loader_kwargs)

        if training and loader_kwargs.get('sampler', None) is None and loader_kwargs.get('balance_method', None) \
                is not None:
            method = loader_kwargs.pop('balance_method')
            assert method.lower() in self.BALANCE_METHODS
            if not hasattr(dataset, 'get_targets'):
                print("Failed to create dataloader with {} balancing. {} does not support `get_targets()`.".format(
                    method, dataset
                ))
            elif method.lower() != 'ldam':
                sampler = balanced_undersampling(dataset) if method.lower() == 'undersample' \
                    else balanced_oversampling(dataset)
                # Shuffle is implied by the balanced sampling
                # loader_kwargs['shuffle'] = None
                loader_kwargs['sampler'] = sampler
            else:
                self.loss = create_ldam_loss(dataset)

        if loader_kwargs.get('sampler', None) is not None:
            loader_kwargs['shuffle'] = None

        # Make sure balance method is not passed to DataLoader at this point.
        loader_kwargs.pop('balance_method', None)

        return DataLoader(dataset, **loader_kwargs)

def balanced_undersampling(dataset, replacement=False):
    tqdm.tqdm.write("Undersampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=replacement)

def balanced_oversampling(dataset, replacement=True):
    tqdm.tqdm.write("Oversampling for balanced distribution.")
    sample_weights, counts = get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=replacement)

def get_label_balance(dataset):
    """
    Given a dataset, return the proportion of each target class and the counts of each class type

    Parameters
    ----------
    dataset

    Returns
    -------
    sample_weights, counts
    """
    assert hasattr(dataset, 'get_targets')
    labels = dataset.get_targets()
    counts = np.bincount(labels)
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    class_freq = counts/counts.sum()
    if len(counts) < 10:
        tqdm.tqdm.write('Class frequency: {}'.format(' | '.join('{:.2f}'.format(c) for c in class_freq)))
    else:
        tqdm.tqdm.write("Class frequencies range from {:.2e} to {:.2e}".format(class_freq.min(), class_freq.max()))
    return sample_weights, counts