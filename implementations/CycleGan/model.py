
from gans.core.models import BaseModel


class CycleGan(BaseModel):
    def on_train_epoch_start(*args, **kwargs):
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
