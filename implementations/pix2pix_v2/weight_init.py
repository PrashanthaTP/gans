from torch import nn
#https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial

def init_weights(net,init='norm',gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(net,'weight') and 'Conv' in classname:
            if init=='norm':
                nn.init.normal_(m.weight.data,mean=0.0,std=gain)
                
            if hasattr(m,'bias') and m.bias is not None :
                nn.init.constant_(m.bias.data,0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data,1.,gain)
            nn.init.constant_(m.bias.data,0.0)
    net.apply(init_func)
    
    return net 

def init_model(model):
    return init_weights(model)


