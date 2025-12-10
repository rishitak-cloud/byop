import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class dataset(Dataset):
    def __init__(self, rootpath, test=False):
        if test:
            self.images = sorted([rootpath+"/DUTS-TE/DUTS-TE-Image/"+ i for i in os.listdir(rootpath+"/DUTS-TE/DUTS-TE-Image/")])
            self.masks = sorted([rootpath+"/DUTS-TE/DUTS-TE-Mask/"+ i for i in os.listdir(rootpath+"/DUTS-TE/DUTS-TE-Mask/")])
        else:
            self.images = sorted([rootpath+"/DUTS-TR/DUTS-TR-Image/"+ i for i in os.listdir(rootpath+"/DUTS-TR/DUTS-TR-Image/")])
            self.masks = sorted([rootpath+"/DUTS-TR/DUTS-TR-Mask/"+ i for i in os.listdir(rootpath+"/DUTS-TR/DUTS-TR-Mask/")])
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index].convert("RGB"))
        mask = Image.open(self.masks[index]).convert("L")
        return self.transform(img), self.transform(mask)
    
    def __len__(self):
        return len(self.images)
