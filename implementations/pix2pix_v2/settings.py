import os
import torch 
import numpy
# from torch.utils.tensorboard.writer import SummaryWriter

from gans.core.utils.config import Config
from gans.core.utils.params import Params
from gans.core.utils.logger import Logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
PARAMS_FILE = os.path.join(BASE_DIR, 'hparams.json')

config = Config.from_json(CONFIG_FILE)
hparams = Params.from_json(PARAMS_FILE)

torch.manual_seed(hparams['seed'])
numpy.random.seed(hparams['seed'])

def get_logger():
    logger = Logger(__name__)
    logger.add_console_handler()
    logger.add_file_handler(log_fullpath=config['log_fullpath'])
    # writer_fake = SummaryWriter(os.path.join(config['runs_dir'], 'fake'), f"{config.version}/real")
    # writer_real = SummaryWriter(os.path.join(config['runs_dir'], 'real'), f"{config.version}/fake")
    # logger.add_writer_real(writer_real)
    # logger.add_writer_fake(writer_fake)

    return logger
