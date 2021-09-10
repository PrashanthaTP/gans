import os
from PIL import Image
from torchvision import datasets,transforms
from torchvision.datasets.folder import IMG_EXTENSIONS 
from torchvision.datasets.vision import VisionDataset


from gans.settings import BASE_DIR
# from torchvision.transforms.functional import to_pil_image




to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()



FILE_DIR = os.path.dirname(os.path.abspath(__file__))
roots = {
    "mnist":f"{BASE_DIR}/datasets",
    "cityscrapes": "E:/Users/VS_Code_Workspace/Machine learning/GANs/dataset/cityscapes/"
    }



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

def get_imgs_list(folder):
    imgs_list = []
    for root,_,files in os.walk(folder):
        for img in sorted(files):
            if img.lower().endswith(IMG_EXTENSIONS):
                imgs_list.append(os.path.join(root,img))
    return imgs_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)

def default_loader(path):
    # from torchvision import get_image_backend
    
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)
    
class Pix2PixFolder(VisionDataset):
    def __init__(self, root, transform,target_transform):
        super().__init__(root ,transform=transform, target_transform = target_transform)
        
        self.root = root 
        self.imgs = get_imgs_list(root)
        self.loader = default_loader
        # self.transform = transform
        # self.target_transform = target_transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        img =  self.loader(self.imgs[idx])
        img_width,_ = img.size
        # print(img.size)
        img = to_tensor(img)
        
        # img = (img-127.5)/127.5 #to make it between -1 to 1
        sample,target = img[:,:,:img_width//2],img[:,:,img_width//2:]
        sample ,target = to_pil(sample),to_pil(target)
        sample = self.transform(sample) if self.transform else sample 
        target = self.target_transform(target) if self.target_transform else target 
        return target,sample
      
    
def get_mnist_dataset(transform,train):
    return datasets.MNIST(root=roots['mnist'],train=train,transform=transform,download=False) 

def get_cityscrapes_dataset(type:str="TRAIN",transform = None,target_transform = None)->Pix2PixFolder:
    """returns Pix2PixFolder

    Args:
        type (str, optional): type of dataset expected. Defaults to "TRAIN".

    Raises:
        ValueError: if type is not in ['TRAIN','TEST']

    Returns:
        Pix2PixFolder: dataset
    """
    if type=="TRAIN":
        return Pix2PixFolder(root=os.path.join(roots['cityscrapes'],'train'),transform = transform,target_transform=target_transform)
    elif type=="TEST":
        return Pix2PixFolder(root=os.path.join(roots['cityscrapes'],'val'),transform=transform,target_transform=target_transform)
    else :
        raise ValueError("type argument can only take values in ['TRAIN','TEST']")