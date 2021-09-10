
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST 
from torchvision import transforms

from gans.core.data.datasets import get_mnist_dataset


def get_dataset(img_size,channels):
 
    transforms_composed =transforms.Compose(
            [transforms.Resize(img_size),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5]*channels,[0.5]*channels
                )]
        )
    return get_mnist_dataset(transforms_composed,train=True)
    

def get_dataloader(hparams):
    dataloader = DataLoader(get_dataset(hparams['img_dim'],
                                         hparams['channels']),
                            batch_size=hparams['batch_size'])
    
    return dataloader