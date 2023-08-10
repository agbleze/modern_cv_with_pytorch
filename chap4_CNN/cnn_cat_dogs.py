#%%
import torchvision
import torch.nn as nn
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch import optim
from torch.utils.data import Dataset, DataLoader

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from torchsummary import summary


#%%
# !pip install -q kaggle

# #%%
# from google.colab import files

# #%%
# files.upload()

# #%%
# !unzip cat-and-dog.zip

# %%
train_data_dir = 'dogs-vs-cats/train'
test_data_dir = 'dogs-vs-cats/test'

#%% move filenames begining with dog to dogs folder
#  !mv dogs-vs-cats/train/dog* dogs-vs-cats/train/dogs;

#%%
#glob(train_data_dir+'/*cat*.jpg')

#%%
#glob(train_data_dir+'/*dog*.jpg')

#%%
class cats_dogs(Dataset):
    def __init__(self, folder):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*jpg')
        self.fpaths = cats + dogs
        
        from random import shuffle, seed
        seed(10)
        self.targets=[fpath.split('/')[-1].startswith('dog')
                      for fpath in self.fpaths
                      ]  ## dog = 1
        
    def __len__(self):
        return len(self.fpaths)
    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224, 224))
        
        return (torch.tensor(im/255).permute(2,0,1)
                .to(device).float(),
                
                torch.tensor([target]).float()
                .to(device)
                )

#%% inspect a random image
data = cats_dogs(train_data_dir)
im, label = data[200]

# %%
plt.imshow(im.permute(1,2,0))
print(label)

# %% define convolution layer
def conv_layer(ni, no, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2)
    )


#%%  define model, loss func optimizer

def get_model():
    model = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

#%%
model, loss_fn, optimizer = get_model()
summary(model, torch.zeros(1, 3, 224, 224))


# %% define get data
def get_data():
    train = cats_dogs(train_data_dir)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val = cats_dogs(test_data_dir)
    val_dl = DataLoader(val, batch_size=32, shuffle=True, drop_last=True)
    return trn_dl, val_dl


#%% define training of batch
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% define accuracy
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

#%% define val loss
@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()



#%% define training process

def trigger_train_process(x, y, model, loss_fn, optimizer,
                          num_epochs: int = 5):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    