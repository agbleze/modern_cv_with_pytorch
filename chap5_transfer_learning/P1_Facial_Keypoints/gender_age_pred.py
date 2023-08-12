
#%%
import torch
import torch.nn as nn
import numpy as np, cv2, pandas as pd, glob, time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms, datasets
from matplotlib import ticker as mticker


#%% fetch dataset




#%%  ##   #####
trn_df = pd.read_csv('fairface-label-train.csv')
val_df = pd.read_csv('fairface-label-val.csv')
trn_df.head()

device = 'cuda' if torch.is_available() else 'cpu'

#%% prepare dataset 
IMAGE_SIZE = 224
class GenderAgeClass(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                              )
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen
    
    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im/255).permute(2,0,1)
        im = self.normalize(im)
        return im[None]
    
    def collate_fn(self, batch):
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)
            ages.append(float(int(age))/80)
            genders.append(float(gender))
        ages, genders = [torch.tensor(x).to(device).float()
                         for x in [ages, genders]
                         ]
        ims = torch.cat(ims).to(device)
        return ims, ages, genders


#%% 
trn = GenderAgeClass(trn_df)
val = GenderAgeClass(val_df)
train_loader = DataLoader(trn, batch_size=32, shuffle=True,
                          drop_last=True, 
                          collate_fn=trn.collate_fn
                          )
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn,)
a,b,c = next(iter(train_loader))
print(a.shape, b.shape, c.shape)

#%%
def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
                            nn.Conv2d(512, 512, kernel_size=3),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Flatten()
                        )
    
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                                        nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.4),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.4),
                                        nn.Linear(128, 64),
                                        nn.ReLU()
                                    )
            self.age_classifier = nn.Sequential(
                                        nn.Linear(64, 1),
                                        nn.Sigmoid()
                                    )
            self.gender_classifier = nn.Sequential(
                                        nn.Linear(64, 1),
                                        nn.Sigmoid()
                                    )
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)




# %%
