import logging
import torch 
from gans.core.engine.base_engine import BaseEngine

class GanEngine(BaseEngine):
    def __init__(self,device:str):
        super().__init__()

        self.curr_epoch = None 
        if device == 'cuda':
            is_cuda_available = torch.cuda.is_available()
            device = torch.device('cuda' if is_cuda_available else 'cpu')
        self.device = torch.device(device)
        
    def __safety_check(self,model,fns):
        for fn_name in fns:
            if not hasattr(model,fn_name):
                raise NotImplementedError(f'{model.__class__.__name__} should have implemented {fn_name}')
            
    def run(self, model,dataloader):
        
        self.__safety_check(model,['train_step'])
        
        logging.debug(f'Training started.(using {self.device})')
 
        
        model.to(self.device)
        for epoch in range(model.hparams.num_epochs):
            self.curr_epoch = epoch
            model.train_epoch(model,dataloader)
            
        
        model.on_train_end()
        logging.debug("Training completed.")
        
    
    def train_epoch(self,model,dataloader):
        for idx,batch in enumerate(dataloader):
            data,targets = batch.data,batch.targets 
            data = data.to(self.device)
            targets = targets.to(self.device)
            model.train_step(data,targets,epoch=self.curr_epoch,batch_idx = idx,device=self.device)
            
            
