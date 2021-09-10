import torch 
from torch import nn ,optim


from gans.core.models.base_model import BaseModel 


class Generator(BaseModel):
    def __init__(self,img_dim,z_dim):
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(z_dim, 256),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(256, img_dim),
                                 nn.Tanh())
        



    def forward(self, x):
        return self.gen(x)
        
    def get_optimizer(self,lr):
        return optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
