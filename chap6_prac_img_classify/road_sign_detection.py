#%%
import os
from torch_snippets import *
from torchvision import transforms as T

#%%
classIds = pd.read_csv('signnames.csv')
classIds.set_index('ClassId', inplace=True)
classIds = classIds.to_dict()['SignName']
classIds = {f'{k:05d}':v for k,v in classIds.items()}
id2int = {v:ix for ix, (k,v) in enumerate(classIds.items())}

#%%
trn_tfms = T.Compose([
    T.ToPILImage(), T.Resize(32), T.CenterCrop(32),
    # T.ColorJitter(brightness=(0.8, 1.2),
    #               contrast=(0.8, 1.2),
    #               saturation=(0.8, 1.2),
    #               hue=0.25 
    #               ),
    
    # T.RandomAffine(5, translate=(0.01, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
])


val_tfms = T.Compose([
    T.ToPILImage(), T.Resize(32), T.CenterCrop(32),T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
])


#%%  
class GTSRB(Dataset):
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
        return img, classIds[clss]
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        imgs, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        classes = [torch.tensor([id2int[clss] for clss in classes])]
        imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
        return imgs, classes


#%%
import torchvision.models as models

def convBlock(ni, no):
    return nn.Sequential(nn.Dropout(0.2),
                         )





