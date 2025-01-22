import torch
import tqdm
import re
import mlflow
from torch_geometric.loader import DataLoader
from sys import gettrace
from .transforms import BatchTransform

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/processes.py
'''
class BaseModelTrainer(object):

    def __init__(self, lr=0.001, l2_weight_decay=0.01, cuda=None, save_model=False, save_model_directory=None, **kwargs):
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
                tqdm.tqdm.write("GPU(s) detected: training and model execution will be performed on GPU.")
            else:
                tqdm.tqdm.write("No GPU detected: training and model execution will be performed on CPU.")
        if isinstance(cuda, bool):
            if cuda:
                cuda = "cuda"
                print(f"Device Properties: {torch.cuda.get_device_properties(cuda)}")
            else:
                cuda = "cpu"

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
        self.save_model = save_model
        self.save_model_dir = save_model_directory

        # TODO: Modify
        self.best_metric = None

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
        batch = next(iterator).to(self.device, non_blocking=self.cuda == 'cuda')
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
    
    def calculate_metrics(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the metrics.

        Returns
        -------
        metrics : dict
                  Dictionary of metrics to be recorded.
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

    def train_step(self, inputs):
        self.train(True)
        outputs = self.forward(inputs)
        loss, recon_loss = self.calculate_loss(inputs, outputs)
        self.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None and self.scheduler_after_batch:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs=outputs, recon_loss=recon_loss)
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
        _, recon_loss = self.calculate_loss(inputs, outputs)


        '''
        NOTE: Currently inputs will be the original signals of each electrode extracted from the graph object
        and the output is the output of the encoder in SPD form. The logits of the outputs are only for the 
        mATT attention module. We will need to improve on this implementation. 
        '''

        metrics = self.calculate_metrics(inputs, outputs=outputs, recon_loss=recon_loss)
        metrics['loss'] = recon_loss
        return metrics

    def predict(self, dataset, **loader_kwargs):
        """
        Determine the outputs for all loaded data from the dataset. This is right now only used in validation (when pretraining is implemented only)

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

        pbar = tqdm.trange(len(dataset), desc="Predicting", ncols=250)
        data_iterator = iter(dataset)

        inputs = list()
        outputs = list()

        with torch.no_grad():
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                output_batch = self.forward(input_batch)

                inputs.append([torch.tensor(tensor).float().cpu() for tensor in input_batch.x])
                if isinstance(output_batch, torch.Tensor):
                    outputs.append(output_batch.cpu())
                else:
                    batch = []
                    for item in output_batch:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                batch.append(v.cpu())
                        else:
                            batch.append(item.cpu())
                    outputs.append(batch)

        ''' 
        TODO: This is very messy code.
        I removed package_multiple_tensors().
        Figure out way to make nicer.  
        ''' 
        
        def package_input(batches):
            result = []
            for b in batches:
                result.append(torch.stack(b, dim=0))
            return result

        def package_output(batches):
            return [torch.cat(b) for b in zip(*batches)]


        return package_input(inputs), package_output(outputs)

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
        tqdm.tqdm.write(start_message)

    def save_best(self):
        """
        Create a snapshot of what is being currently trained for re-loading with the load_best() method.

        Returns
        -------
        best : Any
               Whatever format is needed for load_best(), will be the argument provided to it.
        """
        return [{k: v.cpu() for k, v in self.__dict__[m].state_dict().items()} for m in self._trainables]

    def load_best(self, best):
        """
        Load the parameters as saved by save_best().

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
    
    def fit(self, training_dataset, validation_dataset=None, epochs=1, batch_size=8, **loader_kwargs):
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataloader = self._make_dataloader(training_dataset, training=True, **loader_kwargs)
        print("Training on {} sample batches.".format(len(training_dataloader)))



        mlflow.start_run()
        mlflow.autolog()
        for epoch in range(epochs):
            self.epoch = epoch
            pbar = tqdm.trange(len(training_dataloader), desc="Epoch {}".format(epoch), ncols=400, position=0, leave=True)
            data_iterator = iter(training_dataloader)
            self.train(True)
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                train_metrics = self.train_step(input_batch)
                pbar.set_postfix(train_metrics)
                mlflow.log_metrics(train_metrics, step=iteration)
            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset, **loader_kwargs)
                self.standard_logging(val_metrics, "End of Epoch")
                self._retain_best(val_metrics, val_metrics, 'loss')
            if self.scheduler is not None and not self.scheduler_after_batch:
                self.scheduler.step()

        if self.save_model:
            import pickle as pkl
            best = self.save_best()
            with open(f'{self.save_model_dir}/model_{mlflow.active_run().info.run_id}.pkl', 'wb+') as f:
                pkl.dump(best, f)
        mlflow.end_run()