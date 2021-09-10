import torch 
from torchvision import utils as vutils

from gans.core.engine import GanEngine


from model import CycleGan
from data import get_dataloaders
def train_model(hparams,config,logger):
    model = CycleGan(hparams,config,logger)
    dataloaders = get_dataloaders()
    engine = GanEngine()
    engine.run(model,dataloaders,log_every=50)
    