import torch
import tqdm
import re
import os
import mlflow
from torch_geometric.loader import DataLoader
from sys import gettrace
from Model.MENDR.mAtt.optimizer import MixOptimizer
from pathlib import Path
from Model.loggingUtil import MENDRLogger
import multiprocessing

'''
Based on:
1. https://github.com/SPOClab-ca/dn3/blob/master/dn3/trainable/processes.py
'''
class BaseModelTrainer(object):
    def __init__(self, optimizer, cfg, cuda, **kwargs):
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

        # Note: I think BENDR's implementation is incorrect:
        # https://github.com/SPOClab-ca/dn3/blob/4d477fe42d3d8ce64f3b790585bfa5c7acb84848/dn3/trainable/processes.py#L85
        # Trainables should be tensors/modules that have grads or are in training mode
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor, torch.nn.Parameter)):
                if isinstance(self.__dict__[member], (torch.Tensor, torch.nn.Parameter)) and not self.__dict__[member].requires_grad:
                    self._trainables.append(member)
                if isinstance(self.__dict__[member], torch.nn.Module) and self.__dict__[member].training:
                    self._trainables.append(member)
                
                self.__dict__[member] = self.__dict__[member].to(self.device)
        print(f"Trainables: {self._trainables}")

        self.optimizer = optimizer
        self.scheduler_after_batch = cfg.training_params.scheduler_after_batch
        self.epoch = None
        self.ckpt_dir = cfg.training_params.ckpt_dir
        self.loaded_from_ckpt = False
        self.train_dataset = None
        self.validation_dataset = None

    def _get_batch(self, iterator):
        batch = next(iterator)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Perform batch normalization across channels 
                # Note all data are float32!
                batch[key] = value.float().to(self.device)
        return batch
    
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
    
    def backward(self, encodings, losses):
        '''
        Calculate gradients and update parameters

        Parameters
        ----------
        encodings : list
                    List of encoding matrices
                    Each encoding matrix is a tensor of shape [Batch, Patches, Channels, Time Steps]
        losses : list
                 List of loss tensors
                 Each loss tensor is a tensor of shape [Batch, Patches, Channels, Time Steps]
        '''
        raise NotImplementedError

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)

    def train_step(self, inputs):
        ''' 
        Calculate gradients and update parameters

        Parameters
        ----------
        inputs : dict
                 Dictionary of inputs to the model
                 Keys are the bands of the input data
                 Values are the input data 
                 Shapes are [Batch, Patches, Channels, Time Steps]
        
        Returns
        -------
        train_metrics : dict
                        Dictionary of metrics to be recorded.
        '''
        raise NotImplementedError
        
    def evaluate_step(self, inputs):
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
        raise NotImplementedError
        
    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        val_seen = False
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'val' in m.lower() and not val_seen:
                val_seen = True
                start_message += "\n    "
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            elif m == 'lr':
                start_message += " {}: {:.3e} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.3f} |".format(m, metrics[m])
        tqdm.tqdm.write(start_message)

    def save_best(self, epoch_ckpt_dir):
        """
        Create a snapshot of what is being currently trained for re-loading with the load_best() method.
        """
        assert epoch_ckpt_dir != None, "Checkpoint Directory is none."
        Path(epoch_ckpt_dir).mkdir(parents=True, exist_ok=True)
        for trainable_member in self._trainables:
            torch.save(self.__dict__[trainable_member].state_dict(), os.path.join(epoch_ckpt_dir, f'{trainable_member}_weights.pth'))

    def load_best(self, epoch_ckpt_dir):
        """
        Load the parameters as saved by save_best().
        """
        assert os.path.exists(epoch_ckpt_dir), "Checkpoint Directory does not exist."
        self.load_from_ckpt(epoch_ckpt_dir)
        
    def load_from_ckpt(self, epoch_ckpt_path):
        for trainable_member in self._trainables:
                module_weight_path = os.path.join(epoch_ckpt_path, f'{trainable_member}_weights.pth') 
                assert os.path.exists(module_weight_path), f"{trainable_member}_weights.pth does not exist"
                self.__dict__[trainable_member].load_state_dict(torch.load(module_weight_path, weights_only=True))
        self.optimizer.scheduler.load_state_dict(torch.load(os.path.join(epoch_ckpt_path,"scheduler.pth"), weights_only=False))
        self.loaded_from_ckpt = True


    def _retain_best(self, epoch_idx : int, metrics_to_check: dict):
        """
        Save model depending on the metrics. This must be implemented in the child class.
        """
        raise NotImplementedError

    '''
    def _retain_best(self, epoch_idx : int, metrics_to_check: dict):
        training_Combined_loss = metrics_to_check[f'total_epoch_training_Combined Riemannian Loss']
        validation_Combined_loss = metrics_to_check[f'total_epoch_validation_Combined Riemannian Loss']
        tqdm.tqdm.write(f"Training Riemannian Loss: {training_Combined_loss} Validation Riemannian Loss: {validation_Combined_loss}")
        if f"total_epoch_validation_Wavelet Loss" in metrics_to_check:
            _training_validation_Wavelet_Loss = metrics_to_check[f'total_epoch_training_Wavelet Loss']
            _validation_Wavelet_Loss = metrics_to_check[f'total_epoch_validation_Wavelet Loss']
            tqdm.tqdm.write(f"Training Wavelet Loss: {_training_validation_Wavelet_Loss} Validation Wavelet Loss: {_validation_Wavelet_Loss}")
        tqdm.tqdm.write(" Retaining checkpoint...")

        epoch_ckpt_dir = f'{self.ckpt_dir}/{mlflow.active_run().info.run_id}_{epoch_idx}_{self.mendr_model.contextualizer_size.upper()}'
        self.save_best(epoch_ckpt_dir)
        torch.save(self.optimizer.scheduler.state_dict(), f'{epoch_ckpt_dir}/scheduler.pth')
        self.load_best(epoch_ckpt_dir)
        # Always save scheduler 
    '''

    def _make_dataloader(self, dataset, cfg, training=False):
        if isinstance(dataset, DataLoader):
            return dataset
        loader_kwargs = dict()
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        loader_kwargs.setdefault('num_workers', cfg.training_params.num_workers)
        loader_kwargs.setdefault('batch_size', cfg.training_params.batch_size)
        loader_kwargs.setdefault('persistent_workers', True)
        loader_kwargs.setdefault('shuffle', training)
        loader_kwargs.setdefault('drop_last', training)
        return DataLoader(dataset, **loader_kwargs)

    def _setup_experiment(self, cfg):
        # We cannot log models to MlFlow due to our custom modules.
        assert self.train_dataset != None, "Train Dataset not specified."
        assert self.validation_dataset != None, "Validation Dataset not specified."

        mlflow.set_experiment(cfg.meta_params.experiment_name)
        training_dataloader = self._make_dataloader(self.train_dataset, cfg, training=True)
        print("Training on {} sample batches.".format(len(training_dataloader)))

        validation_dataloader = None
        if self.validation_dataset != None:
            validation_dataloader = self._make_dataloader(self.validation_dataset, cfg,training=False)
            print("Validation on {} sample batches.".format(len(validation_dataloader)))

        if "mlflow_run_id" in cfg.meta_params:
            mlflow.start_run(run_id=cfg.meta_params.mlflow_run_id,
                            run_name=cfg.meta_params.run_name,
                            experiment_name=cfg.meta_params.experiment_name,
                            log_system_metrics=cfg.meta_params.log_system_metrics)
        else:
            mlflow.start_run(run_name=cfg.meta_params.run_name,
            log_system_metrics=cfg.meta_params.log_system_metrics)

        if "log_model_params_and_grads" in cfg.meta_params and cfg.meta_params.log_model_params_and_grads:
            self.logger = MENDRLogger()

        return training_dataloader, validation_dataloader
    
    def _epoch_metrics(self, aggregated_metrics, metric_dict, step):
        for metric in metric_dict:
            if metric != 'lr':
                if f'total_epoch_{step}_{metric}' not in aggregated_metrics:
                    aggregated_metrics[f'total_epoch_{step}_{metric}'] = metric_dict[metric]
                else:
                    aggregated_metrics[f'total_epoch_{step}_{metric}'] += metric_dict[metric]
        return aggregated_metrics

    
    def fit(self, training_dataset, cfg, validation_dataset=None):
        """
        Fit the specific model to the training dataset. This must be implemented in the child class.
        """
        raise NotImplementedError