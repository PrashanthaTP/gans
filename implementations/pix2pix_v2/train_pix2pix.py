from engine import Engine 
from model import Pix2PixModel
from data import get_dataloader




def train(hparams,config,logger):
    try:
        train_dataloader = get_dataloader(config['train_dataset_dir'],hparams['train_batch_size'],for_train=True)
        val_dataloader = get_dataloader(config['val_dataset_dir'],hparams['val_batch_size'],for_train=False)
        engine = Engine()
        model = Pix2PixModel(hparams,config,logger)
        engine.run(model,train_dataloader,val_dataloader,['gen_loss','disc_loss'])
        
    except Exception as e:
        logger.exception(e)