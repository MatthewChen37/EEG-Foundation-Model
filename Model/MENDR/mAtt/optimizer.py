from .utils import *
from . import StiefelParameter
import concurrent.futures
from functools import partial

'''
Modified from https://github.com/CECNL/MAtt/blob/main/mAtt/optimizer.py
'''
class MixOptimizer(object):
    """This is a meta optimizer which uses other optimizers for updating parameters
        and remap all StiefelParameter parameters to Stiefel space after they have been updated.
    """
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def set_scheduler_T_max(self, T_max):
        print(f'Set Cosine Annealing optimizer T_max to: {T_max}')
        self.scheduler.T_max = T_max

    def set_scheduler_t0(self, T_0):
        print(f'Set Cosine Annealing with Warm Restarts optimizer T_0 to: {T_0}')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=2, eta_min=1e-7)

    def scheduler_step_cosine_annealing(self):
        self.scheduler.step()

    def scheduler_step_cosine_annealing_warm_restarts(self, iteration):
        self.scheduler.step(iteration)

    def step(self, closure=None):
        """Performs a single optimization step with parallel processing of parameters."""
        
        # Pre-processing phase: Store parameter states and prepare gradients
        def process_param_before_step(p):
            if p.grad is None:
                return None
            
            if isinstance(p, StiefelParameter):
                # Store the current parameter state
                param_state = p.data.clone()
                
                # Perform orthogonal projection
                p.data.fill_(0)
                trans = orthogonal_projection(p.grad.data, p.data)
                p.grad.data.fill_(0).add_(trans)
                
                return (id(p), param_state)
            return None
        
        # Create parameter lists for processing
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group['params'])
        
        # Process parameters in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_param_before_step, all_params))
        
        # Update the state dictionary
        for result in results:
            if result is not None:
                param_id, param_state = result
                if param_id not in self.state:
                    self.state[param_id] = param_state
                else:
                    self.state[param_id].fill_(0).add_(param_state)
        
        # Perform the optimizer step
        loss = self.optimizer.step(closure)
        
        # Post-processing phase: Map parameters back to Stiefel space
        def process_param_after_step(p):
            if p.grad is None:
                return
            
            if isinstance(p, StiefelParameter):
                # Perform retraction to map back to Stiefel space
                trans = retraction(p.data, self.state[id(p)])
                p.data.fill_(0).add_(trans)
        
        # Process parameters in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_param_after_step, all_params)
        return loss
    
    '''
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        temp_dict = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)
                    
                    p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)
        return loss
    '''
