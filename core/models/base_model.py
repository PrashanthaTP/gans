import torch 
from torch import nn 

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
     
    def save(self,state_dict,checkpoint_path):
  
        torch.save(state_dict,checkpoint_path)
        logger.debug('Model saved to {checkpoint_path} ')
        
    def load(self,checkpoint_path):
       self.load_state_dict(torch.load(checkpoint_path))
    
    def forward(self,*args,**kwargs):
        raise NotImplementedError(f'All classes inheriting from BaseModel must implement forward method.')
    
  
    def  on_train_epoch_start(*args,**kwargs):
        raise NotImplementedError() 
    
    def  on_eval_epoch_start(*args,**kwargs):
        raise NotImplementedError() 
    
    def  on_train_epoch_end(*args,**kwargs):
        raise NotImplementedError() 
    
    def  on_eval_epoch_end(*args,**kwargs):
        raise NotImplementedError() 
    
    
    def train_step(*args,**kwargs):
        raise NotImplementedError() 
    
    def eval_step(*args,**kwargs):
        raise NotImplementedError() 
    
    def on_run_start(*args,**kwargs):
        raise NotImplementedError() 
    
    def on_run_end(*args,**kwargs):
        raise NotImplementedError() 