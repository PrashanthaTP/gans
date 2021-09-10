import itertools
import os
import torch 
from torchvision import transforms
from matplotlib import pyplot as plt

from nets.generator import Generator

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
resize = transforms.Resize(512)

def get_generator(hparams):
    gen = Generator(hparams['img_channels'], hparams['ngf'])
  
    gen.weight_init(mean=0.0, std=0.02)
   

    return gen
def test(test_dataloader,hparams,config,gen_checkpoint):
    gen = get_generator(hparams)
    gen.load(gen_checkpoint)
    gen.eval()
    with torch.no_grad():
        img_dir = config['imgs_dir']
        img_name = f'pix2pix_out.png'
        test_x,test_y = next(iter(test_dataloader))
        test_x,test_y = test_x[:5],test_y[:5]

        fake = gen(test_x)

        nrows = test_x.shape[0]
        ncols = 3
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))

        for i, j in itertools.product(range(nrows), range(ncols)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for i in range(nrows):

            ax[i, 0].cla()
            img = to_pil(test_x[i])
            img = resize(img)
            ax[i, 0].imshow(img.cpu().data.numpy().transpose(1, 2, 0))
            
            ax[i, 1].cla()
            img = to_pil(test_y[i])
            img = resize(img)
            ax[i, 1].imshow(img.cpu().data.numpy().transpose(1, 2, 0))
            
            ax[i, 2].cla()
            img = to_pil(fake[i])
            img = resize(img)
            ax[i, 2].imshow(img.cpu().data.numpy().transpose(1, 2, 0)+1)

        titles = ["Input Image", "Ground Truth", "Predicted Image"]

        for i in range(ncols):
            ax[0, i].set_title(titles[i])

        label = f'Pix2Pix : Cityscrapes dataset'
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(os.path.join(img_dir, img_name))
