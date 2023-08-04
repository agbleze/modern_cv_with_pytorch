#%%
import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam


#%% ### GET DATASET ###
data_folder = 'data/fmnist/'
data = datasets.FashionMNIST(data_folder, train=True, download=True)

tr_images = data.data
tr_targets = data.targets

data_val = datasets.FashionMNIST(data_folder, train=False, download=True)
val_images = data_val.data
val_targets = data_val.targets


#%% ## define class for retrieving data ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class FashionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        x = self.x[ix]
        y = self.y[ix]
        self.x = torch.tensor(x.to(device))
        self.y = torch.tensor(y.to(device))
        return self.x, self.y
    
    
#%% define class for loading batches of data

def get_data():
    tr_data = FashionDataset(tr_images, tr_targets)
    tr_dl = DataLoader(tr_data, batch_size=32, shuffle=True)
    
    val_data = FashionDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images), shuffle=False)
    return tr_dl, val_dl


#%% define model, loss_fn, opt


        
        





