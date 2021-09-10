import os
import torch 
import unittest

from torch.serialization import save

from data import get_dataloader
from nets.discriminator import PatchDiscriminator as Discriminator
from nets.generator import Generator

from utils import save_images 

dataset_dir = r'E:////Users////VS_Code_Workspace////Machine learning////GANs////dataset////archive////tiny-imagenet-200////train'
def get_batch():
    dl = get_dataloader(dataset_dir, 16, for_train=True)
    # print(len(dl.dataset))
    return  next(iter(dl))
    
    
class Test(unittest.TestCase):
    def test_discriminator(self):
        x = torch.randn(1,3,256,256) 
        y = torch.randn(1,3,256,256) 
        model = Discriminator(in_channels = 3)
        out = model(x,y)
        print(f"Discriminator out shape : {out.shape}")
        self.assertEqual(out.shape,(1,1,26,26))
        
    def test_generator(self):
        x = torch.randn(1,2,256,256)
        model = Generator()
        out = model(x)
        print(f"Generator's output shape : ",x.shape )
        self.assertEqual(out.shape,(1,3,256,256))
        
    def test_dataloader(self):
       batch = get_batch()
       L,ab = batch
       print(f"Batch size : ",L.shape[0])
       print(f"Shapes of L and ab")
       print(L.shape,ab.shape)
       self.assertEqual(L.shape,(L.shape[0],1,256,256))
       self.assertEqual(ab.shape,(L.shape[0],2,256,256))
       
    def test_images(self):
        batch_size= 5
        # L = torch.rand((batch_size,1,256,256))
        # real_ab = torch.rand((batch_size,2,256,256))
        # fake_ab = torch.rand((batch_size,2,256,256))
        batch = get_batch()
        L,real_ab,fake_ab = batch[0][:5],batch[1][:5],batch[1][:5]
    
        # for each in [L,real_ab,fake_ab]:
        #     each = each/255
        epoch = 0
        config = {'imgs_dir':os.path.dirname(os.path.abspath(__file__))}
        save_images(L, real_ab, fake_ab, epoch, config)
        
def main():
    suite = unittest.TestSuite()
    suite.addTest(Test('test_images'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    return
if __name__=='__main__':
    main()
