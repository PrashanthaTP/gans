import torch 
from torch import nn,optim 
from torch.nn import functional as F 


from gans.core.models.base_model import BaseModel


#TODO => modify to resemble https://www.tensorflow.org/tutorials/generative/pix2pix#input_pipeline
class Generator(BaseModel):
    def __init__(self,in_channels,ngf):
        super().__init__()
        
        # Unet Encoder
        self.conv1 =  nn.Conv2d(in_channels,ngf,4,2,1)
        self.conv2 = self.encoder_block(ngf,ngf*2,4,2,1,add_batchnorm=True)
        self.conv3 = self.encoder_block(ngf*2,ngf*4,4,2,1,add_batchnorm=True)
        self.conv4 = self.encoder_block(ngf*4,ngf*8,4,2,1,add_batchnorm=True)
        
        self.conv5 = self.encoder_block(ngf*8,ngf*8,4,2,1,add_batchnorm=True)
        self.conv6 = self.encoder_block(ngf*8,ngf*8,4,2,1,add_batchnorm=True)
        self.conv7 = self.encoder_block(ngf*8,ngf*8,4,2,1,add_batchnorm=True)
        
        self.conv8 = self.encoder_block(ngf*8,ngf*8,4,2,1,add_batchnorm=False)
        # Unet decoder
        self.deconv1 = self.decoder_block(ngf*8,ngf*8,4,2,1)
        self.deconv2 = self.decoder_block(ngf*8*2,ngf*8,4,2,1)
        self.deconv3 = self.decoder_block(ngf*8*2,ngf*8,4,2,1)
        self.deconv4 = self.decoder_block(ngf*8*2,ngf*8,4,2,1)
        
        self.deconv5 = self.decoder_block(ngf*8*2,ngf*4,4,2,1)
        self.deconv6 = self.decoder_block(ngf*4*2,ngf*2,4,2,1)
        self.deconv7 = self.decoder_block(ngf*2*2,ngf,4,2,1)
        self.deconv8 = nn.Sequential(nn.ReLU(),nn.ConvTranspose2d(ngf*2,in_channels,4,2,1))
        # # Unet Encoder
        # self.conv1 =  nn.Conv2d(in_channels,ngf,4,2,1)
        # self.conv2 = self.get_encoder_block(ngf,ngf*2,4,2,1)
        # self.conv3 = self.get_encoder_block(ngf*2,ngf*4,4,2,1)
        # self.conv4 = self.get_encoder_block(ngf*4,ngf*8,4,2,1)
        
        # self.conv5 = self.get_encoder_block(ngf*8,ngf*8,4,2,1)
        # self.conv6 = self.get_encoder_block(ngf*8,ngf*8,4,2,1)
        # self.conv7 = self.get_encoder_block(ngf*8,ngf*8,4,2,1)
        
        # self.conv8 = nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.ReLU())
        # # Unet decoder
        # self.deconv1 = self.get_decoder_block(ngf*8,ngf*8,4,2,1)
        # self.deconv2 = self.get_decoder_block(ngf*8*2,ngf*8,4,2,1)
        # self.deconv3 = self.get_decoder_block(ngf*8*2,ngf*8,4,2,1)
        # self.deconv4 = self.get_decoder_block(ngf*8*2,ngf*8,4,2,1)
        
        # self.deconv5 = self.get_decoder_block(ngf*8*2,ngf*4,4,2,1)
        # self.deconv6 = self.get_decoder_block(ngf*4*2,ngf*2,4,2,1)
        # self.deconv7 = self.get_decoder_block(ngf*2*2,ngf,4,2,1)
        # self.deconv8 = nn.Sequential(nn.ConvTranspose2d(ngf*2,in_channels,4,2,1),nn.Tanh())
        
        

        
    def get_encoder_block(self,in_channels,out_channels,kernel_size,stride,padding,lrelu_slope=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(lrelu_slope)
            )
    
    def get_decoder_block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        
    def encoder_block(self,in_c,out_c,kernel_size=4,stride=2,padding=1,add_batchnorm=False):
        if add_batchnorm:
            return nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_c,out_c,kernel_size,stride,padding),
                nn.BatchNorm2d(out_c)
                )
        else:
            return  nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(in_c,out_c,kernel_size,stride,padding),
                )
            
    def decoder_block(self,in_c,out_c,kernel_size=4,stride=2,padding=1):

            return nn.Sequential(
                
                nn.ReLU(),
                nn.ConvTranspose2d(in_c,out_c,kernel_size,stride,padding),
                nn.BatchNorm2d(out_c)
                )
    
    def get_optimizer(self,lr=0.0002,b1 = 0.5,b2 = 0.999):
        return optim.Adam(self.parameters(),lr=lr,betas=(b1,b2))
    
    def forward(self,x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)
        
        x = F.dropout(self.deconv1(e8),0.5,training=True)
        # d1 = torch.cat([d1,e7],1)
        x = F.dropout(self.deconv2(torch.cat([x,e7],1)),0.5,training=True)
        # d2 = torch.cat([d2,e6],1)
        x = F.dropout(self.deconv3(torch.cat([x,e6],1)),0.5,training=True)
        # d3 = torch.cat([d3,e5],1)
        
        x = self.deconv4(torch.cat([x, e5], 1))
        # d4 = torch.cat([d4,e4],1)
        x = self.deconv5(torch.cat([x, e4], 1))
        # d5 = torch.cat([d5,e3],1)
        x = self.deconv6( torch.cat([x, e3], 1))
        # d6 = torch.cat([d6,e2],1)
        x = self.deconv7(torch.cat([x, e2], 1))
        # d7 = torch.cat([d7,e1],1)
    
        return torch.tanh(self.deconv8(torch.cat([x, e1], 1)))

      
    
        
      
    def weight_init(self,mean,std):
        for m in self._modules:
            self.__normal_init(self._modules[m],mean,std)
            
    def __normal_init(self,m,mean,std):
        if isinstance(m,(nn.ConvTranspose2d,nn.Conv2d)):
            m.weight.data.normal_(mean,std)
            m.bias.data.zero_()
