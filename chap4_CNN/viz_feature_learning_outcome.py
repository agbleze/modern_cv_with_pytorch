
#%% download dataste
!wget https://www.dropbox.com/s/5jh4hpuk2gcxaaq/all.zip
!unzip all.zip

# %%
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam
from torchvision import datasets
import numpy as np, cv2
import matplotlib.pyplot as plt
from glob import glob
from imgaug import augmenters as iaa
from torchsummary import summary

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tfm = iaa.Sequential(iaa.Resize(28))

#%% class takes folder as input and loops through files in path

class XO(Dataset):
    def __init__(self, folder):
        self.files = glob(folder) # loops through files in path
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, ix):
        f = self.files[ix]
        im = tfm.augment_image(cv2.imread(f))[:,:,0]
        im = im[None] # create a dumpy channel at the beginning of img shape
        cl = f.split('/')[-1].split('@')[0]=='x'
        return (torch.tensor(1-im/255).to(device).float(), 
                torch.tensor([cl]).float().to(device)
                )
        
        
#%%
data = XO('all/*')

#%%
R, C = 7, 7
fig, ax = plt.subplots(R, C, figsize=(5,5))
for label_cases, plot_row in enumerate(ax):
    for plot_cell in plot_row:
        plot_cell.grid(False); plot_cell.axis('off')
        ix = np.random.choice(1000)
        im, label = data[ix]
        print()
        plot_cell.imshow(im[0].cpu(), cmap='gray')
plt.tight_layout()


#%% define model
def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3200, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ).to(device)
    
    loss_fn = nn.BCELoss()
    optimizer =Adam(params=model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

#%% summarize model architecture
model, loss_fn, optimizer = get_model()
summary(model, torch.zeros(1,1,28,28))    


# %%
