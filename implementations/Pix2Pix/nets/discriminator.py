import torch 
from torch import nn,optim 
from functools import partial
from gans.core.models.base_model import BaseModel 


class Discriminator(BaseModel):
    def __init__(self,ndf):
        super().__init__()
        
        partial_disc_block_s1 = partial(self.disc_block,kernel_size=4,stride=1,padding=1)
        partial_disc_block_s2 = partial(self.disc_block,kernel_size=4,stride=2,padding=1)
        
        self.disc = nn.Sequential(
                nn.Conv2d(6,ndf,4,2,1),
                nn.LeakyReLU(),
                
                partial_disc_block_s2(in_channels=ndf,out_channels=ndf*2),
                partial_disc_block_s2(in_channels=ndf*2,out_channels=ndf*4),
                
                partial_disc_block_s1(in_channels=ndf*4,out_channels=ndf*8),
                
                nn.Conv2d(ndf*8,1,4,1,1),
                # nn.Sigmoid()
            )
    
    def disc_block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels)      ,
            nn.LeakyReLU(0.2)
            )
         
    def get_optimizer(self,lr=0.0002,b1 = 0.5,b2 = 0.999):
        return optim.Adam(self.parameters(),lr=lr,betas=(b1,b2))  
    
    def forward(self,x,labels):
        x = torch.cat([x,labels],1)
        return self.disc(x)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            self.__normal_init(self._modules[m], mean, std)

    def __normal_init(self,m, mean, std):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
