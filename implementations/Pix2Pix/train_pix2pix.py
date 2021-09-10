import itertools
import os 
import time 
import torch 
from torch import nn 
from torchvision import utils as vutils
from matplotlib import pyplot as plt

from nets.generator import Generator 
from nets.discriminator import Discriminator 



def setup_models(hparams):
    gen =  Generator(hparams['img_channels'],hparams['ngf'])
    disc = Discriminator(hparams['ndf'])
    gen.weight_init(mean=0.0,std=0.02)
    disc.weight_init(mean=0.0,std=0.02)

    return gen,disc
    
def train_model(dataloader,hparams,config,logger):
    device_str = 'cpu'
    if hparams['device']=='cuda':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device=device_str)
    gen,disc = setup_models(hparams)
    gen.to(device)
    disc.to(device)
    
    gen_optim = gen.get_optimizer(lr=hparams['lr'],b1=hparams['beta1'],b2=hparams['beta2'])
    disc_optim = disc.get_optimizer(lr=hparams['lr'],b1=hparams['beta1'],b2=hparams['beta2'])
    criterion = nn.BCELoss()
    disc_criterion = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()
    steps = 0
    num_epochs = hparams['num_epochs']
    
    test_x,test_y = next(iter(dataloader.test))
    
    test_x,test_y = test_x[:5],test_y[:5]
    scaler = torch.cuda.amp.GradScaler()
    logger.info(f"Traing started on {device}")
    start_time = time.time()
    for epoch in range(num_epochs):
        gen.train()
        disc.train()
        disc_losses = []
        gen_losses = []
        epoch_start = time.time()
        for x,y in dataloader.train: 
            x = x.to(device)
            y = y.to(device)
            #########################
            # Discriminator training
            ########################
            with torch.cuda.amp.autocast():
                disc_out = disc(x,y).squeeze() #removes dimension having value 1
                disc_real_loss = disc_criterion(disc_out,torch.ones_like(disc_out))
            
                fake = gen(x)
                disc_out = disc(x,fake.detach()).squeeze()
                disc_fake_loss = disc_criterion(disc_out,torch.zeros_like(disc_out))
            
                disc_loss = (disc_real_loss + disc_fake_loss)/2
            
            disc_optim.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optim)
            # disc_optim.step()
            
            disc_losses.append(disc_loss.item())
            #########################
            # Generator training
            ########################
            with torch.cuda.amp.autocast():
                disc_out = disc(x,fake).squeeze()
            
                gen_loss = disc_criterion(disc_out,torch.ones_like(disc_out)) + 100*(L1_loss(fake,y))
            
            gen_optim.zero_grad()
            scaler.scale(gen_loss).backward()
            scaler.step(gen_optim)
            # gen_optim.step()
            scaler.update()
            gen_losses.append(gen_loss.item())
            
            steps += 1
            if steps %50==0:
                logger.add_scalars_to_board(loss_gen=gen_loss.item(),loss_disc=disc_loss.item(),step=steps)
                
        epoch_end = time.time()
        epoch_time = epoch_end-epoch_start
        
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] : {epoch_time:0.2f} seconds or {epoch_time/60:0.2f} minutes : loss_d: {sum(disc_losses)/len(disc_losses):0.4f} : loss_g: {sum(gen_losses)/len(gen_losses):0.4f}")
        
        validate(gen,test_x,test_y,epoch,config,logger,device)
        
        gen.save(gen.state_dict(), config['gen_checkpoint_fullpath'])
        disc.save(disc.state_dict(),config['disc_checkpoint_fullpath'])
    hparams.save(config['hparams_save_fullpath'])
    
    end_time = time.time()
    train_time = end_time-start_time 
    logger.info(f"Training completed successfully")
    logger.info(f"Training took {train_time:0.2f} seconds or {train_time/60:0.2f} minutes.")
        
#TODO add labels to each column in the plot:DONE
def validate(gen,test_x,test_y,epoch,config,logger,device):
    gen.eval()
    img_dir = config['imgs_dir']
    img_name = f'pix2pix_out_{epoch}.png'
    # x,y = next(iter(test_dataloader))
    # x,y = x[:5],y[:5]
    # x = x.to(device)
    with torch.no_grad():
        test_x = test_x.to(device)
        fake  = gen(test_x)

        nrows = test_x.shape[0]
        ncols = 3
        fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(5,5))
    
        for i,j in itertools.product(range(nrows),range(ncols)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)

            
            
        for i in range(nrows):
                
            ax[i,0].cla()
            ax[i,0].imshow((test_x[i].cpu().data.numpy().transpose(1,2,0)+1)/2)# make images lie between 0 to 1
            ax[i,1].cla()
            ax[i,1].imshow((test_y[i].cpu().data.numpy().transpose(1,2,0)+1)/2)
            ax[i,2].cla()
            ax[i,2].imshow((fake[i].cpu().data.numpy().transpose(1,2,0)+1)/2)
            
        titles = ["Input Image","Ground Truth","Predicted Image"]
        
        for i in range(ncols):
            ax[0, i].set_title(titles[i],fontsize=10)
        
        label = f'Epoch {epoch+1}'
        fig.text(0.5,0.04,label,ha='center')
        plt.savefig(os.path.join(img_dir,img_name))
        plt.close(fig)
  
    # logger.writers['fake_writer'].add_image("Pix2Pix/Cityscrapes",fig,step=epoch)
