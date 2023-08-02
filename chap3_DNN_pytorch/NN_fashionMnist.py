#%%
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
data_folder = 'data/fmnist'

#%% get data

fmnist = datasets.FashionMNIST(data_folder, train=True)

tr_images = fmnist.data
tr_targets = fmnist.targets


#%% create class for dataset to fecth it

class FMINISTDataset(Dataset):
    def __init__(self, x, y) -> None:
        x = x.float()
        x = x.view(-1, 28*28)
        self.x, self.y = x, y
        
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)
    
    def __len__(self) -> int:
        return len(self.x)
        


#%% create a function to generate training dataloader
def get_data():
    train = FMINISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl









