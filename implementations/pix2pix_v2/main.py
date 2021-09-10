from train_pix2pix import train
from settings import config,hparams, get_logger 
def main():
    logger = get_logger()
    train(hparams,config,logger)
if __name__=='__main__':
    main()