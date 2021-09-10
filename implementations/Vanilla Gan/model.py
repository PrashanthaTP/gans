import torch 
import os
from torch import nn,optim
from torchvision import utils as vutils 

from nets.generator import Generator 
from nets.discriminator import Discriminator

import time 


class VanillaGan:
    def __init__(self,hparams,config,logger):
        self.hparams = hparams
        self.config = config 
        self.logger = logger 
        
        self.gen = Generator(hparams['img_dim']**2,hparams['z_dim']) 
        self.disc = Discriminator(hparams['img_dim']**2)
        self.criterion = nn.BCELoss()
        
        device = self.hparams['device']
        if device == 'cuda':
            is_cuda_available = torch.cuda.is_available()
            device = torch.device('cuda' if is_cuda_available else 'cpu')
        self.device = torch.device(device)
       
        
    def get_noise(self,size):
        return torch.randn(size=size)
    

    def fit(self,dataloader):
        fixed_noise = torch.randn((32,self.hparams['z_dim'])).to(self.device)
        self.gen.to(self.device)
        self.disc.to(self.device)
        disc_optim = self.disc.get_optimizer(self.hparams['lr'])
        gen_optim = self.gen.get_optimizer(self.hparams['lr'])
        self.logger.info(f'training started on {self.device}')
        step = 0
        num_epochs = self.hparams['num_epochs']
        start = time.time()
        for epoch in range(num_epochs):
            for batch_idx,(real,_) in enumerate(dataloader):
                z_dim = self.hparams['z_dim']
                real = real.view(-1, self.hparams['img_dim']**2).to(self.device)
               
                batch_size = real.shape[0]

                ##################################
                # Descriminator Training
                # maximize log(D(real)) + log(1-D(G(Z)))
                ##################################
                noise = torch.randn(batch_size, z_dim).to(self.device)
                # to make the gradients persist, call detach while passing to generator
                fake = self.gen(noise)
                disc_real = self.disc(real).view(-1)
                loss_d_real = self.criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = self.disc(fake.detach()).view(-1)
                loss_d_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_d = (loss_d_real + loss_d_fake)/2
                disc_optim.zero_grad()
                loss_d.backward()
                # loss_d_fake.backward(retain_graph=True)
                disc_optim.step()

                #################################
                # Generator Training
                # minimize log(1-D(G(z))) <-> max log(D(G(z)))
                #################################
                out = self.disc(fake).view(-1)
                loss_g = self.criterion(out, torch.ones_like(out))
                self.gen.zero_grad()
                loss_g.backward()
                gen_optim.step()

                step += 1
                if step%20==0:
                     self.add_scalars_to_board(loss_gen=loss_g,loss_disc=loss_d,step=step)
                     
                if batch_idx==len(dataloader)-2:
                    with torch.no_grad():
                        self.gen.eval()
                        img_dim = self.hparams['img_dim']
                        channels = self.hparams['channels']
                        fake_imgs = self.gen(fixed_noise).reshape(-1, channels, img_dim,img_dim)
                        real_imgs = real.reshape(-1, channels, img_dim,img_dim)
                        
                        self.add_imgs_to_board(real_imgs,fake_imgs,epoch)
                        
            self.logger.info(f"Epoch [{epoch+1}/{num_epochs}] gen_loss:{loss_g.item():0.4f} disc_loss:{loss_d.item():0.4f}")
        end = time.time()
        time_taken = end-start 
        self.logger.info('Training successfully completed.')
        self.logger.info(f"Training took {time_taken:0.2f} seconds or {time_taken/60:0.2f} minutes.")
        self.on_train_end()
           
    def add_scalars_to_board(self,loss_gen,loss_disc,step):
        writer_fake,writer_real = self.logger.writers['writer_fake'],self.logger.writers['writer_real']  
        writer_real.add_scalar('loss/disc', loss_disc.item(), global_step=step)
        writer_fake.add_scalar('loss/gen', loss_gen.item(), global_step=step)
        
    def add_imgs_to_board(self,real_imgs,fake_imgs,step):
        img_grid_fake = vutils.make_grid(fake_imgs, normalize=True)
        img_grid_real = vutils.make_grid(real_imgs, normalize=True)
        writer_fake,writer_real = self.logger.writers['writer_fake'],self.logger.writers['writer_real']  
        writer_fake.add_image("MNIST Images/fake",img_grid_fake, global_step=step)
        writer_real.add_image("MNIST Images/real",img_grid_real, global_step=step)
        
        vutils.save_image(fake_imgs,os.path.join(self.config['imgs_dir'],f'gen_img_{step}.png'))
        

    def on_train_end(self):
        self.gen.save(self.gen.state_dict(),self.config['gen_checkpoint_fullpath'])
        self.disc.save(self.disc.state_dict(),self.config['disc_checkpoint_fullpath'])
        self.hparams.save(self.config['hparams_save_fullpath'])
