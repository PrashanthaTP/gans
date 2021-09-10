import torch 
import os 

from torch.utils.tensorboard.writer import SummaryWriter

from gans.core.utils.params import Params 
from gans.core.utils.config import Config 
from gans.core.utils.logger import Logger
from gans.core.utils.gif_maker import create_gif
 


from data import get_dataloader
from train_pix2pix import train_model

#############################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
PARAMS_FILE = os.path.join(BASE_DIR, 'hparams.json')
#############################################

config = Config.from_json(os.path.join(BASE_DIR, 'config.json'))
hparams = Params.from_json(PARAMS_FILE)
#######################################
# To get similar results on every run 
torch.manual_seed(hparams['seed'])
#######################################

def get_logger():
    logger = Logger(__name__)
    logger.add_console_handler()
    logger.add_file_handler(log_fullpath=config['log_fullpath'])
    writer_fake = SummaryWriter(os.path.join(config['runs_dir'],'fake'),f"{config.version}/real")
    writer_real = SummaryWriter(os.path.join(config['runs_dir'],'real'),f"{config.version}/fake")
    logger.add_writer_real(writer_real)
    logger.add_writer_fake(writer_fake)

    return logger 

def train():
    logger = get_logger()
    dataloader = get_dataloader(hparams) 

    train_model(dataloader,config=config,hparams=hparams,logger=logger)
    


if __name__ == '__main__':
    train()
    imgs_dir = config['imgs_dir']
    # images_dir = images_dir%({"version":config.version})
    gif_path = config['gif_file_fullpath']
    # gif_path = r'E:\Users\VS_Code_Workspace\Python\VirtualEnvironments\gans\gans\gifs\pix2pix\Run_4_3_2021__21_44\pix2pix_out_Run_4_3_2021__21_44.gif'
    
    create_gif(imgs_dir,gif_path)
    print(f"gif created .")
