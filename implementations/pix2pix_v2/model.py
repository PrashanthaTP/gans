import torch 
from torch.cuda import amp
from torch import nn, optim

import time 

from nets.generator import Generator
from nets.discriminator import PatchDiscriminator

from losses import GanLoss
from weight_init import init_weights
from utils import save_images

# torch.manual_seed(10)
class Pix2PixModel(nn.Module):
    def __init__(self, hparams, config, logger) -> None:
        super().__init__()
        self.hparams = hparams
        self.config = config
        self.logger = logger
        
        self.gen = Generator(hparams)
        self.disc = PatchDiscriminator(hparams)
        self.gen = init_weights(self.gen)
        self.disc = init_weights(self.disc)

        self.device = torch.device('cpu')
        self.device = torch.device(
            'cuda') if self.hparams['device'] == 'cuda' and torch.cuda.is_available() else self.device

        self.criterion = GanLoss(device=self.device)
        self.L1_loss = nn.L1Loss()
        self.optim_gen = self.gen.optimizer
        self.optim_disc = self.disc.optimizer

        self.scaler = amp.GradScaler()
        
        

    def forward(self, x):
        return self.gen(x)

    def on_train_start(self):
        self.logger.info(f"Pix2Pix : version : {self.config.version}")
        self.logger.debug(f"Training started on {self.device}")
        self.gen.to(self.device)
        self.disc.to(self.device)
        self.epoch_times = []
        
    def on_epoch_start(self,state):
        self.epoch_start_time = time.time()
        
    def train_step(self, batch, state):
        logs = {}
        self.gen.train()
        self.disc.train()

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        ###############################
        # Discriminator training
        # fake = G(x)
        # Maximize ylog(D(x,y)) + (1-y)log(1-(D(x,G(x))))
        ###############################
        with amp.autocast():
            fake = self.gen(x)
            loss_disc_real = self.criterion(self.disc(x, y), is_real_targets=True)
            loss_disc_fake = self.criterion(self.disc(x, fake.detach()), is_real_targets=False)
            loss_disc = (loss_disc_real+loss_disc_fake)/2

        self.optim_disc.zero_grad()
        self.scaler.scale(loss_disc).backward()
        self.scaler.step(self.optim_disc)

        ###############################
        # Generator training
        # fake = G(x)
        # Maximize ylog(D(x,G(x)))
        ###############################
        with amp.autocast():
            loss_gen = self.criterion(self.disc(x, fake), is_real_targets=True)
            loss_gen += self.L1_loss(fake,y)*self.hparams['l1_lambda']
        self.optim_gen.zero_grad()
        self.scaler.scale(loss_gen).backward()
        self.scaler.step(self.optim_gen)

        self.scaler.update()

        logs['loss'] = {}
        logs['loss']['gen_loss'] = loss_gen.item()
        logs['loss']['disc_loss'] = loss_disc.item()
        return logs

    def __get_log_string(self, losses):

        log = ''
        for key, value in losses.items():
            log += f"| {key} : {value:0.4f} "
        return log

    def log(self, losses, state):
        # log = self.__get_log_string(losses)
        # self.logger.info(log)
        pass

    def validation_step(self, val_dl, state):
        self.gen.eval()
        batch = next(iter(val_dl))
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        fake = self.gen(x)
        save_images(x,y,fake,state['epoch'],self.config)
        
    def on_epoch_end(self, losses, state):
        duration = time.time()-self.epoch_start_time
        self.epoch_times.append(duration)
        curr_epoch = state['epoch']
        num_epochs = self.hparams['num_epochs']
        self.logger.info(f' [Epoch {curr_epoch+1}/{num_epochs}][ {duration:0.2f} seconds (or) {duration/60:0.2f} minutes] ' + self.__get_log_string(losses))
        
    
        torch.save(self.gen.state_dict(),self.config['gen_checkpoint_fullpath'])
        torch.save(self.disc.state_dict(),self.config['disc_checkpoint_fullpath'])
        self.hparams.save(self.config['hparams_fullpath'])

    def on_train_end(self):
        log = "Training completed successfully\n"
        duration = sum(self.epoch_times)
        log += f"Total time taken {duration:0.2f} seconds or {duration/60:0.2f} minutes."
        self.logger.info(log)
