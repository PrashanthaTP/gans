import torch

from collections import namedtuple 

from torch.utils.data import DataLoader
from torchvision import transforms
from gans.core.data.datasets import get_cityscrapes_dataset


def resize_images(imgs,resize_scale=256):
    outputs = torch.FloatTensor(imgs.shape[0],imgs.shape[1],resize_scale,resize_scale)
    resize = transforms.Resize(resize_scale)
    to_PIL = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i in range(imgs.shape[0]):
        outputs[i] = to_tensor(resize(to_PIL(imgs[i])))
    
    return outputs

# def custom_resize(resize_scale):
#     def resize(img):
#         resize = transforms.Resize(resize_scale)
        
        
loader_tuple = namedtuple('loaders',('train','test'))
def get_dataloader(params):
    transforms_composed = transforms.Compose(
            [ 
            transforms.Resize(params['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5]*params['img_channels'],std=[0.5]*params['img_channels'])
            ]
        )
    train_loader= DataLoader(get_cityscrapes_dataset("TRAIN",transform=transforms_composed,target_transform=transforms_composed),batch_size=params["batch_size"],shuffle=True)
    test_loader = DataLoader(get_cityscrapes_dataset("TEST",transform=transforms_composed,target_transform=transforms_composed),batch_size=params["batch_size"],shuffle=True)
    
    return loader_tuple(train_loader,test_loader)
    # test = iter(test_loader).next()[0] #channels,height,width
    # img_width = test.shape[2]//2

    
    # test_x,test_y = test[:,:,:,img_width:],test[:,:,:,:img_width]
    # test_x,test_y = resize_images(test_x),resize_images(test_y)
    
    # test_x,test_y = test[:,:,:,img_width:],test[:,:,:,:img_width]
    # test_x,test_y = resize_images(test_x),resize_images(test_y)
    
if __name__ == '__main__':
    get_dataloader({'batch_size':10,'num_channels':3,'img_size':128})
