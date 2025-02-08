import torch
import tqdm
import re
import os
import mlflow
from torch_geometric.loader import DataLoader
from sys import gettrace
from .transforms import BatchTransform
from Model.MENDR.mAtt.optimizer import MixOptimizer
from pathlib import Path

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/processes.py
'''
class BaseModelTrainer(object):

    def __init__(self, lr=0.001, l2_weight_decay=0.01, cuda=None, ckpt_dir=None, **kwargs):
        """
        By default uses the SGD with momentum optimization.
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

        # Names of trainable objects
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
                    self._trainables.append(member)
                self.__dict__[member] = self.__dict__[member].to(self.device)
        print(f"Trainables: {self._trainables}")
        self.optimizer = MixOptimizer(torch.optim.SGD(self.parameters(), weight_decay=l2_weight_decay, lr=lr, nesterov=True, momentum=0.9))
        self.scheduler_after_batch = True
        self.epoch = None
        self.lr = lr
        self.weight_decay = l2_weight_decay
        self.ckpt_dir = ckpt_dir
        self.best_metric = float("Inf")

    def set_optimizer(self, optimizer):
        # assert isinstance(optimizer, torch.optim.Optimizer)
        del self.optimizer
        self.optimizer = optimizer
        self.lr = float(self.optimizer.optimizer.param_groups[0]['lr'])

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
        batch = next(iterator)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Perform batch normalization across channels 
                batch[key] = self._std_norm(value.float().to(self.device))
        return batch
    
    def _std_norm(self, x):
        mean = torch.mean(x, dim=(0, 1), keepdim=True)
        std = torch.std(x, dim=(0, 1), keepdim=True)
        x = (x - mean) / std
        return x


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
            #print(f"{member}: {sum(p.numel() for p in self.__dict__[member].parameters() if p.requires_grad)}")


    def forward(self, data):
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

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)
            if hasattr(member, 'freeze_features'):
                member.freeze_features(unfreeze=mode)

    def train_step(self, inputs):
        self.train(True)
        outputs = self.forward(inputs)

        encoder_output = outputs['encoder_output']
        combined_r2e_output = outputs['combined_r2e_output']
        combined_manifold_output = outputs['combined_manifold_output']
        wavelet_r2e_output = outputs['wavelet_r2e_output']
        wavelet_manifold_output = outputs['wavelet_manifold_output']
        combined_loss = outputs['combined_loss']
        wavelet_loss = outputs['wavelet_loss']
        combined_acc = outputs['combined_acc']
        wavelet_acc = outputs['wavelet_acc']
        recon_loss, loss_dict = self.reconstruction_loss(inputs, encoder_output)
        recon_loss = recon_loss
        total_loss = recon_loss + combined_loss + wavelet_loss
        self.backward(total_loss)
        self.optimizer.step()
        train_metrics = self.calculate_metrics(combined_loss.item(), wavelet_loss.item(), recon_loss.item(), combined_acc, wavelet_acc)
        train_metrics.setdefault('loss', total_loss.item())
        train_metrics["LR"] = str(self.optimizer.scheduler.get_last_lr()[0])
        for band, loss in loss_dict.items():
            train_metrics[f'{band} Loss'] = loss
        if self.scheduler_after_batch:
            self.optimizer.scheduler_step(loss)
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
        eval_metrics = self.predict(dataset, **loader_kwargs)
        return eval_metrics

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
        dataset = self._make_dataloader(dataset, **loader_kwargs)

        pbar = tqdm.trange(len(dataset), desc="Predicting", ncols=250)
        data_iterator = iter(dataset)

        combined_loss_agg = 0
        wavelet_loss_agg = 0
        recon_loss_agg = 0
        combined_acc_agg = 0
        wavelet_acc_agg = 0
        with torch.no_grad():
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                outputs = self.forward(input_batch)

                encoder_output = outputs['encoder_output']
                combined_r2e_output = outputs['combined_r2e_output']
                combined_manifold_output = outputs['combined_manifold_output']
                wavelet_r2e_output = outputs['wavelet_r2e_output']
                wavelet_manifold_output = outputs['wavelet_manifold_output']
                combined_loss_agg += outputs['combined_loss'].item()
                wavelet_loss_agg += outputs['wavelet_loss'].item()
                combined_acc_agg += outputs['combined_acc']
                wavelet_acc_agg += outputs['wavelet_acc']

                recon_loss, loss_dict = self.reconstruction_loss(input_batch, encoder_output)
                recon_loss_agg += recon_loss.item()

            total_loss_agg = combined_loss_agg + wavelet_loss_agg + recon_loss_agg
            val_metrics = self.calculate_metrics(combined_loss_agg, wavelet_loss_agg, recon_loss_agg, combined_acc_agg / len(pbar), wavelet_acc_agg / len(pbar))
            val_metrics.setdefault('loss', total_loss_agg)
            return val_metrics

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
        assert self.ckpt_dir != None, "Checkpoint Directory is none."
        Path(f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}').mkdir(parents=True, exist_ok=True)
        run_save_dir = f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}'
        for trainable_member in self._trainables:
            torch.save(self.__dict__[trainable_member].state_dict(), os.path.join(run_save_dir, f'{trainable_member}_weights.pth'))
    def load_best(self):
        """
        Load the parameters as saved by save_best().

        """
        ckpt_dir = f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}'
        assert os.path.exists(ckpt_dir), "Checkpoint Directory does not exist."
        for trainable_member in self._trainables:
                module_weight_path = os.path.join(ckpt_dir, f'{trainable_member}_weights.pth') 
                assert os.path.exists(module_weight_path), f"{trainable_member}_weights.pth does not exist"
                self.__dict__[trainable_member].load_state_dict(torch.load(module_weight_path))

    def _retain_best(self, metrics_to_check: dict):
        if not os.path.exists(f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}') or metrics_to_check['loss'] <= self.best_metric:
            tqdm.tqdm.write(f"Best loss: {metrics_to_check['loss']}. Retaining checkpoint...")
            self.best_metric = metrics_to_check['loss']
            self.save_best()
        else:
            tqdm.tqdm.write(f"Failed to beat best loss. Curr Loss: {metrics_to_check['loss']}. Best Loss: {self.best_metric}. Reverting to old checkpoint...")
            self.load_best()

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
        signature = None
        for epoch in range(epochs):
            epoch_metrics = {
                'total_epoch_training_loss': 0,
                'total_epoch_combined_training_loss': 0,
                'total_epoch_wavelet_training_loss': 0,
                'total_epoch_reconstruction_training_loss': 0,
                'total_epoch_validation_loss': 0,
                'total_epoch_combined_validation_loss': 0,
                'total_epoch_wavelet_validation_loss': 0,
                'total_epoch_reconstruction_validation_loss': 0,
            }
            self.epoch = epoch
            pbar = tqdm.trange(len(training_dataloader), desc="Epoch {}".format(epoch), ncols=400, position=0, leave=True)
            data_iterator = iter(training_dataloader)
            self.train(True)
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                train_metrics = self.train_step(input_batch)
                epoch_metrics['total_epoch_training_loss'] += train_metrics['loss']
                epoch_metrics['total_epoch_combined_training_loss'] += train_metrics['Combined Loss']
                epoch_metrics['total_epoch_wavelet_training_loss'] += train_metrics['Wavelet Loss']
                epoch_metrics['total_epoch_reconstruction_training_loss'] += train_metrics['Recon Loss']
                pbar.set_postfix(train_metrics)
                mlflow.log_metrics(train_metrics, step=epoch*len(pbar) + iteration)
            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset, **loader_kwargs)
                epoch_metrics['total_epoch_validation_loss'] += val_metrics['loss']
                epoch_metrics['total_epoch_combined_validation_loss'] += val_metrics['Combined Loss']
                epoch_metrics['total_epoch_wavelet_validation_loss'] += val_metrics['Wavelet Loss']
                epoch_metrics['total_epoch_reconstruction_validation_loss'] += val_metrics['Recon Loss']
                self.standard_logging(val_metrics, "End of Epoch")
                self._retain_best(val_metrics)
                mlflow.log_metrics(val_metrics, step=epoch * len(pbar) + iteration)
            if not self.scheduler_after_batch:
                self.optimizer.scheduler_step(val_metrics['loss'])
            print("Epoch: ", epoch, "Total Training Loss: ", epoch_metrics['total_epoch_training_loss'], "Total Validation Loss: ", epoch_metrics['total_epoch_validation_loss'])
            mlflow.log_metrics(epoch_metrics, step=epoch)

        if self.ckpt_dir != None:
            print(f"Saved Model to: {self.ckpt_dir/{mlflow.active_run().info.run_id}}")
        mlflow.end_run()