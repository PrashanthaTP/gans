import os
from gans.core.utils.config import Config 
from gans.core.utils.params import Params

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
PARAMS_FILE = os.path.join(BASE_DIR, 'hparams.json')

config = Config.from_json(CONFIG_FILE)
hparams = Params.from_json(PARAMS_FILE)

