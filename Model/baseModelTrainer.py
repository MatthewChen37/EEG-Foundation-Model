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

        # Names of trainable objects
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor, torch.nn.Parameter)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
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

    def calculate_metrics(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the metrics.

        Returns
        -------
        metrics : dict
                  Dictionary of metrics to be recorded.
        """
        raise NotImplementedError

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)
            if hasattr(member, 'freeze_features'):
                member.freeze_features(unfreeze=mode)

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
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |" + "\n"
        for m in metrics:
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m]) + "\n"
            elif m == 'lr':
                start_message += " {}: {:.3e} |".format(m, metrics[m]) + "\n"
            else:
                start_message += " {}: {:.3f} |".format(m, metrics[m]) + "\n"
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
            mlflow.start_run(run_id=cfg.meta_params.mlflow_run_id, run_name=cfg.meta_params.run_name, experiment_name=cfg.meta_params.experiment_name)
        else:
            mlflow.start_run(run_name=cfg.meta_params.run_name)
        self.logger = MENDRLogger()

        if self.loaded_from_ckpt == False:
            self.optimizer.set_scheduler_t0(len(training_dataloader))
        return training_dataloader, validation_dataloader

    
    def fit(self, training_dataset, cfg, validation_dataset=None):
        """
        Fit the specific model to the training dataset. This must be implemented in the child class.
        """
        raise NotImplementedError

    '''
    def fit(self, training_dataset, validation_dataset=None, epochs=1, batch_size=8, mlflow_run_id=None, **loader_kwargs):
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataloader = self._make_dataloader(training_dataset, training=True, **loader_kwargs)
        print("Training on {} sample batches.".format(len(training_dataloader)))

        validation_dataloader = None
        if validation_dataset != None:
            validation_dataloader = self._make_dataloader(validation_dataset, training=False, **loader_kwargs)
            print("Validation on {} sample batches.".format(len(validation_dataloader)))

        if mlflow_run_id != None:
            mlflow.start_run(run_id=mlflow_run_id)
        else:
            mlflow.start_run()
        self.logger = MENDRLogger()

        if self.loaded_from_ckpt == False:
            self.optimizer.set_scheduler_t0(len(training_dataloader))
        for epoch in range(epochs):
            epoch_metrics = {}
            self.epoch = epoch

    '''
''' TRAINING '''
    
'''
train_pbar = tqdm.trange(len(training_dataloader), desc="Epoch {}".format(epoch), ncols=400, position=0, leave=True)
train_data_iterator = iter(training_dataloader)
self.train(True)
for iteration in train_pbar:
    input_batch = self._get_batch(train_data_iterator)
    train_metrics = self.train_step(input_batch)
    train_pbar.set_postfix(train_metrics)
    mlflow.log_metrics(train_metrics, step=epoch*len(train_pbar) + iteration)
    epoch_metrics = self._epoch_metrics(epoch_metrics, train_metrics, "training")
    if self.scheduler_after_batch:
        self.optimizer.scheduler_step(epoch*len(train_pbar) + iteration)
    # Logging
    self.logger.log_model_gradients(self.mendr_model.mendr_encoder, epoch=epoch * len(train_pbar) + iteration)
    self.logger.log_model_gradients(self.mendr_model.mendr_contextualizer, epoch=epoch * len(train_pbar) + iteration)
    if self.mendr_model.contextualizer_size.upper() == "LARGE":
        self.logger.log_model_gradients(self.mendr_model.mendr_contextualizer.temp1, epoch=epoch * len(train_pbar) + iteration, name="Temperature")
        self.logger.log_model_gradients(self.mendr_model.mendr_contextualizer.CombinedContextualizer.mask, epoch=epoch * len(train_pbar) + iteration, name="Mask")
    elif self.mendr_model.contextualizer_size.upper() == "TINY":
        self.logger.log_model_gradients(self.mendr_model.mendr_contextualizer.Contextualizer.mask, epoch=epoch * len(train_pbar) + iteration, name="Mask")
    else:
        raise ValueError("Unidentified Contextualizer Type")

# VALIDATION 
'''
'''
if validation_dataloader != None:
self.train(False)
pbar = tqdm.trange(len(validation_dataloader), desc="Predicting", ncols=400, position=0, leave=True)
val_data_iterator = iter(validation_dataloader)
for iteration in pbar:
    input_batch = self._get_batch(val_data_iterator)
    val_metrics = self.evaluate_step(input_batch, iteration)
    epoch_metrics = self._epoch_metrics(epoch_metrics, val_metrics, "validation")
    pbar.set_postfix(val_metrics)
'''
# ''' SAVE '''
'''
self._retain_best(epoch, epoch_metrics)
self.standard_logging(epoch_metrics, "End of Epoch")
self.logger.logEncoderParams(self.mendr_model.mendr_encoder, step=epoch)
self.logger.logContextualizerParams(self.mendr_model.mendr_contextualizer, step=epoch)
if self.mendr_model.contextualizer_size.upper() == 'LARGE':
    self.logger.logMENDRTrainerParams(self.mendr_model.mendr_contextualizer.temp1, self.mendr_model.mendr_contextualizer.CombinedContextualizer.mask, step=epoch)
elif self.mendr_model.contextualizer_size.upper() == 'TINY':
    self.logger.logMENDRTrainerParams(None, self.mendr_model.mendr_contextualizer.Contextualizer.mask, step=epoch)
else:
    raise ValueError("Unidentified Contextualizer Type")
mlflow.log_metrics(epoch_metrics, step=epoch)
print("Epoch: ", epoch, "Total Training Loss: ", epoch_metrics['total_epoch_training_Combined Riemannian Loss'], "Total Validation Loss: ", epoch_metrics['total_epoch_validation_Combined Riemannian Loss'])
if self.ckpt_dir != None:
        print(f"Saved Model to: {self.ckpt_dir}/{mlflow.active_run().info.run_id}_{self.epoch}_{self.mendr_model.contextualizer_size.upper()}")
if not self.scheduler_after_batch:
    self.optimizer.scheduler_step(epoch)

mlflow.end_run()
self.logger.closeWriter()
'''
'''
def _epoch_metrics(self, aggregated_metrics, metric_dict, step):
    for metric in metric_dict:
        if metric != 'lr':
            if metric not in aggregated_metrics :
                aggregated_metrics[f'total_epoch_{step}_{metric}'] = metric_dict[metric]
            else:
                aggregated_metrics[f'total_epoch_{step}_{metric}'] += metric_dict[metric]
    return aggregated_metrics
'''