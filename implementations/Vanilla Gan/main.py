import torch 
import os 

from torch.utils.tensorboard.writer import SummaryWriter

from gans.core.utils.params import Params 
from gans.core.utils.config import Config 
from gans.core.utils.logger import Logger
from gans.core.utils.gif_maker import create_gif
 

from model import VanillaGan 
from data import get_dataloader

#############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR,'config.json')
PARAMS_FILE = os.path.join(BASE_DIR,'hparams.json')
#############################################

config = Config.from_json(os.path.join(BASE_DIR, 'config.json'))
hparams = Params.from_json(PARAMS_FILE)
#######################################
# To get similar results on every run 
torch.manual_seed(hparams['seed'])
#######################################

logger = Logger(__name__)
logger.add_console_handler()
logger.add_file_handler(log_fullpath=config['log_fullpath'])
writer_fake = SummaryWriter(os.path.join(config['runs_dir'],'fake'),f"{config.version}/real")
writer_real = SummaryWriter(os.path.join(config['runs_dir'],'real'),f"{config.version}/fake")
# logger.add_writer('writer_real',writer_real)
# logger.add_writer('writer_fake',writer_fake)
logger.add_writer_real(writer_real)
logger.add_writer_fake(writer_fake)
def train():
    dataloader = get_dataloader(hparams) 
    model = VanillaGan(hparams,config,logger)
    model.fit(dataloader)
    


if __name__ == '__main__':
    train()
    create_gif(config['imgs_dir'],config['gif_file_fullpath'])
    print(f"gif created .")
