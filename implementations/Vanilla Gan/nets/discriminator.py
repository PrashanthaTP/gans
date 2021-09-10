from torch import nn,optim 

from gans.core.models.base_model import BaseModel
class Discriminator(BaseModel):
    def __init__(self, img_dim):
        super().__init__()
      
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid())
        
    def forward(self, x):
    
        return self.disc(x)
    def get_optimizer(self,lr):
        return optim.Adam(self.disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
