import numpy as np
import os
from PIL import Image
from collections import namedtuple
from torchvision import transforms as vtransforms
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2lab, lab2rgb
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

Sample = namedtuple('sample',('L','ab'))

class MapDataset(Dataset):
    def __init__(self, root_dir, has_classes=True):
        self.root_dir = root_dir
        self.files = []
        if has_classes:
            dirs = sorted([entry.name for entry in os.scandir(self.root_dir) if entry.is_dir()])
            for dir in dirs:
                dir_path = os.path.join(self.root_dir, dir, 'images')
                image_files = sorted([os.path.join(dir_path,entry.name) for entry in os.scandir(dir_path) 
                                      if entry.is_file() and entry.name.lower().endswith(IMG_EXTENSIONS)])
                self.files.extend(image_files[:20])
        else:
            self.files =  sorted([os.path.join(self.root_dir,entry.name) for entry in os.scandir(self.root_dir) 
                             if entry.is_file() and entry.name.lower().endswith(IMG_EXTENSIONS)])[:20]
        
        self.transforms = vtransforms.Compose(
              [  vtransforms.Resize((256,256),interpolation=Image.BICUBIC)]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transforms(img)
    
        return split_image(img)


def split_image(img):
    img = np.array(img)
    # img = img.transpose(1,2,0)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = vtransforms.ToTensor()(img_lab)
    L = img_lab[[0], ...]/50 - 1  # -1- to 1 after selecting first channel
    ab = img_lab[[1, 2], ...]/110  # -1 to 1
    return Sample(L,ab)


def get_dataloader(dataset_root_dir, batch_size,for_train):
    dataset = MapDataset(dataset_root_dir,has_classes=for_train)
    return DataLoader(dataset, batch_size,shuffle=True)
