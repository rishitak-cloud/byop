import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import albumentations as A
import numpy as np

def train_transform():
    return A.Compose([
    A.SmallestMaxSize(max_size=128 * 2, p=1.0),
    A.RandomCrop(height=128, width=128, p=1.0),
    A.SquareSymmetry(p=1.0), 
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

def val_transform():
    return A.Compose([
    A.SmallestMaxSize(max_size=128 * 2, p=1.0),
    A.CenterCrop(height=128 * 2, width=128 * 2, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

class dataset(Dataset):
    def __init__(self, rootpath, transform=None, test=False):
        if test:
            self.images = sorted([rootpath+"/DUTS-TE/DUTS-TE-Image/"+ i for i in os.listdir(rootpath+"/DUTS-TE/DUTS-TE-Image/")])
            self.masks = sorted([rootpath+"/DUTS-TE/DUTS-TE-Mask/"+ i for i in os.listdir(rootpath+"/DUTS-TE/DUTS-TE-Mask/")])
            self.transform = transform
        else:
            self.images = sorted([rootpath+"/DUTS-TR/DUTS-TR-Image/"+ i for i in os.listdir(rootpath+"/DUTS-TR/DUTS-TR-Image/")])
            self.masks = sorted([rootpath+"/DUTS-TR/DUTS-TR-Mask/"+ i for i in os.listdir(rootpath+"/DUTS-TR/DUTS-TR-Mask/")])
        #self.transform = transforms.Compose([
         #   transforms.Resize((128,128), interpolation=InterpolationMode.NEAREST),
         #   transforms.ToTensor()
        #])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        img_np = np.array(img, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.int64)
        if self.transform:
            augmented = self.transform(img_np, mask_np)
            return augmented['image'], augmented['mask']
        return img_np, mask_np
    
    def __len__(self):
        return len(self.images)

class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, mask = self.subset[index]
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            return augmented['image'], augmented['mask']
            
        return img, mask

    def __len__(self):
        return len(self.subset)