
#%% class acctivation maps
import os
from torch_snippets import *
from torchvision import transforms as T

#%% indices of output classes
id2int = {'Parasitized': 0, 'Uninfected': 1}

#%% transformations done on images
trn_tfms = T.Compose([
                        T.ToPILImage(),
                        T.Resize(128),
                        T.CenterCrop(128),
                        T.ColorJitter(brightness=(0.95, 1.05),
                                    contrast=(0.95,1.05),
                                    saturation=(0.95,1.05),
                                    hue=0.05
                                    ),
                        T.RandomAffine(5, translate=(0.01,0.1)),
                        T.ToTensor(), 
                        T.Normalize(mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5]
                                    )
                ])

# transformation for val dataset
val_tfms = T.Compose([
                        T.ToPILImage(),
                        T.Resize(128),
                        T.CenterCrop(128),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                    ])


#%% define the dataset class
class MalariaImages(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        logger.info(len(self))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, ix):
        fpath = self.files[ix]
        clss = fname(parent(fpath))
        img = read(fpath, 1)
        return img, clss
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        pass

