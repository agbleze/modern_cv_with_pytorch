
#%% import modules
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torchsummary import summary
from PIL import Image
from torch import optim
import cv2, numpy as np, pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt

#%%  #######
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% load vgg16 model
model = models.vgg16(pretrained=True).to(device)
summary(model, torch.zeros(1,3,224,224))



#%% ####### implement VGG16 for image classification  ######
train_data_dir = 'training_set/training_set'
test_data_dir = 'test_set/test_set'

#%%
class CatsDogs(Dataset):
    def __init__(self, folder, num_images=500):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*.jpg')
        if not num_images:
            self.fpaths = cats + dogs
        else:
            self.fpaths = cats[:num_images] + dogs[:num_images]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                              )
        from random import seed, shuffle
        seed(10); shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog')
                        for fpath in self.fpaths
                        ]
        
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224,224))
        im = torch.tensor(im/255)
        im = im.permute(2,0,1)
        im = self.normalize(im)
        return (im.float().to(device), 
                torch.tensor([target]).float().to(device)
                )


#%% fetch image and labels
data = CatsDogs(train_data_dir)

#%%
im, label = data[200]
plt.imshow(im.permute(1,2,0).cpu())
print(label)

#%% download vgg16
def get_model():
    model = models.vgg16(pretrained=True)
    # freeze params in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # replace avgpool module with feature map of 1 X 1
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    
    # define classifier module
    model.classifier = nn.Sequential(nn.Flatten(),
                                     nn.Linear(512, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(128, 1),
                                     nn.Sigmoid()
                                    )
    
    # define loss function
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer

#%% ### summary of model
model, criterion, optimizer = get_model()
summary(model, torch.zeros(1, 3, 224, 224))

# %%
