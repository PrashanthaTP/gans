import torch 
from torch import nn,optim 


class Block(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,act='relu',use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
           nn.Conv2d(in_channels,out_channels,4,2,1,bias=False,padding_mode='reflect')
           if down==True else nn.ConvTranspose2d(in_channels,out_channels,4,2,1,bias=False),
           nn.BatchNorm2d(out_channels) ,
           nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
           )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class Generator(nn.Module):
    def __init__(self,hparams ,features=64):
        super().__init__()
        self.hparams = hparams  
        gen_in_channels = self.hparams['gen_in_channels']
        gen_out_channels = self.hparams['gen_out_channels']
        self.initial_down = nn.Sequential(
            nn.Conv2d(gen_in_channels,features,4,2,1,padding_mode='reflect'),#in_channels is one as input is in black and white or 'L' in LAB format
            nn.LeakyReLU(0.2)
            )
        
        self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False)
        
        self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False)
        
        self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False)
        
        self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        
        self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        
        self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False)
        
        self.bottleneck = nn.Sequential( 
                                nn.Conv2d(features*8,features*8,4,2,1,padding_mode="reflect"),
                                nn.ReLU()
                                         )
        
        self.up1 = Block(features*8,features*8,down=False,act="relu",use_dropout=True)
        self.up2 = Block(features*8*2,features*8,down=False,act="relu",use_dropout=True)
        self.up3 = Block(features*8*2,features*8,down=False,act="relu",use_dropout=True)
        self.up4 = Block(features*8*2,features*8,down=False,act="relu",use_dropout=False)
        self.up5 = Block(features*8*2,features*4,down=False,act="relu",use_dropout=False)
        self.up6 = Block(features*4*2,features*2,down=False,act="relu",use_dropout=False)
        self.up7 = Block(features*2*2,features,down=False,act="relu",use_dropout=False)

        self.final_up = nn.Sequential( 
                             nn.ConvTranspose2d(features*2,gen_out_channels,4,2,1),nn.Tanh()       #output channels is 2 == a*b* in LAB format 
                                      )
        
        
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        x = self.up1(bottleneck)
        x= self.up2(torch.cat([x,d7],dim=1))
        x = self.up3(torch.cat([x,d6],dim=1))
        x = self.up4(torch.cat([x,d5],dim=1))
        x = self.up5(torch.cat([x,d4],dim=1))
        x = self.up6(torch.cat([x,d3],dim=1))
        x = self.up7(torch.cat([x,d2],dim=1))
        return self.final_up(torch.cat([x,d1],dim=1))
     
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(),lr=self.hparams ['lr'],betas=(0.5,0.999))
        