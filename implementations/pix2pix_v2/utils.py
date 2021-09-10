import itertools
import matplotlib.pyplot as plt 
import torch 
import os
import numpy as np 
from skimage.color import lab2rgb

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    # print(torch.max(L),torch.max(ab))
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def save_images(L,real_ab,fake_ab,epoch,config): 

    img_dir = config['imgs_dir']
    img_name = f'pix2pix_out_{epoch}.png'
    L,real_ab,fake_ab = L.detach(),real_ab.detach(),fake_ab.detach()
    
    x = lab_to_rgb(L,real_ab)
    y = lab_to_rgb(L,fake_ab)
    def test():
        nonlocal x 
        print([np.max(x[i,:,:,i]) for i in range(3)],x.shape)
        t= x[0]
        print("max",np.mean([np.max(t[:,:,i],axis=0) for i in range(3)]))
        print("min",np.mean([np.min(t[:,:,i],axis=0) for i in range(3)]))
        print("mean",np.mean([np.mean(t[:,:,i],axis=0) for i in range(3)]))
        print("std",np.mean([np.std(t[:,:,i],axis=0) for i in range(3)]))
    # test()
    
    nrows = x.shape[0]
    ncols = 3
    fig, ax = plt.subplots(nrows=nrows, 
                           ncols=ncols,
                           gridspec_kw=dict(wspace=0.1, hspace=0.1,
                                            top=1 - 0.5  / (nrows + 1), bottom=0.5  / (nrows + 1),
                                            left=0.5 / (ncols + 1), right=1 - 0.5 / (ncols + 1)
                                            ),
                           figsize=(ncols + 1, nrows+ 1),
                           sharey='row', sharex='col')

    for i, j in itertools.product(range(nrows), range(ncols)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(nrows):
        ax[i, 0].cla()
        ax[i, 0].imshow(L[i].cpu().numpy().transpose(1,2,0),cmap="gray")
        ax[i, 1].cla()
        ax[i, 1].imshow(x[i])
        ax[i, 2].cla()
        ax[i, 2].imshow(y[i])

    titles = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(ncols):
        ax[0, i].set_title(titles[i], fontsize=8)

    label = f'Epoch {epoch+1}'
    fig.text(0.5, 0.04, label, ha='center',fontsize=8)
    # plt.title(label,y=1.08)
    # plt.tight_layout(0.5)
    # plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(img_dir, img_name))
    plt.close(fig)

    # fig = plt.figure(figsize=(15, 8))
    # for i in range(8):
    #     ax = plt.subplot(4, 5, i + 1)
    #     ax.imshow(L[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
    #     ax.axis("off")
    #     ax = plt.subplot(4, 5, i + 1 + 5)
    #     ax.imshow(x[i])
    #     ax.axis("off")
    #     ax = plt.subplot(4, 5, i + 1 + 10)
    #     ax.imshow(y[i])
    #     ax.axis("off")
    # plt.show()
    
    # logger.writers['fake_writer'].add_image("Pix2Pix/Cityscrapes",fig,step=epoch)
