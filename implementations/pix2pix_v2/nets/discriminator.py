import torch
from torch import nn,optim 

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,padding_mode='reflect',bias=False),#bias False as BatchNorm isekf has a bias component
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) 
            )
        
    def forward(self,x):
        return self.conv(x)
    
    

class PatchDiscriminator(nn.Module):
    def __init__(self,hparams,features=(64,128,256,512)): 
        """Input of 289x289 images ==>30x30 patch

        Args:
            in_channels (int): [description]
            features (tuple, optional): [description]. Defaults to (64,128,256,512).
        """
        super().__init__()
        self.hparams = hparams 
        in_channels = self.hparams['disc_in_channels'] 
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels,features[0],4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
            )
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(ConvBlock(in_channels,feature,stride=1 if feature==features[-1] else 2))
            in_channels = feature 
        
        layers.append(
            nn.Conv2d(in_channels,1,4,1,1,padding_mode="reflect")
            )
        self.model = nn.Sequential(*layers)
    
    @property
    def optimizer(self):
        return optim.Adam(self.parameters(),lr=self.hparams['lr'],betas=(0.5,0.999))
        
    def forward(self,x,y):
        x = torch.cat([x,y],dim=1)
        x = self.initial(x)
        return self.model(x)


        