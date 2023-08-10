
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


# %% define func for training and return loss and accuracy
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item(), is_correct[0]

#%% define DataLoader
trn_dl = DataLoader(XO('all/*'), batch_size=32, drop_last=True)

#%%
model, loss_fn, optimizer = get_model()

#%% train model over 5 epochs
for epoch in range(5):
    print(epoch)
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x,y, model, loss_fn, optimizer)


#%% fetch an image to check what the filters learn about the image
im, c = trn_dl.dataset[2]
plt.imshow(im[0].cpu())
plt.show()

#%% pass image through trained model and fetch output 
## of 1st layer
first_layer = nn.Sequential(*list(model.children())[:1])
intermediate_output = first_layer(im[None])[0].detach()

#%% plot output of 64 filters
fig, ax = plt.subplots(8, 8, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.set_title('Filter: '+str(ix))
    axis.imshow(intermediate_output[ix].cpu())
plt.tight_layout()
plt.show()


# %% fetch multiple img
x, y = next(iter(trn_dl))
x2 = x[y==0]
# %% reshape to have a proper shape for CNN
x2 = x2.view(-1,1,28,28)

first_layer = nn.Sequential(*list(model.children())[:1])
first_layer_output = first_layer(x2).detach()
n = 4
fig, ax = plt.subplots(n, n, figsize=(10,10))

for ix, axis in enumerate(ax.flat):
    axis.imshow(first_layer_output[ix, 4, :, :].cpu())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()


#%% create model that extracts layers until 2nd convolution layer
second_layer = nn.Sequential(*list(model.children())[:4])
second_intermediate_output = second_layer(im[None])[0].detach()

#%% plot output
fig, ax = plt.subplots(11,11, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(second_intermediate_output[ix].cpu())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()

# %% 34th filter's output
second_layer = nn.Sequential(*list(model.children())[:4])
second_intermediate_output = second_layer(x2).detach()
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(second_intermediate_output[ix,34,:,:].cpu())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()


#%% ######## plot activations of fully connected layer  ######

custom_dl = DataLoader(XO('all/*'), batch_size=2498, drop_last=True)
x,y = next(iter(custom_dl))
x2 = x[y==0]
x2 = x2.view(len(x2), 1, 28, 28)

#%%
flatten_layer = nn.Sequential(*list(model.children())[:7])
flatten_layer_output = flatten_layer(x2).detach()

#%% ### plot the flattened layer
plt.figure(figsize=(100, 10))
plt.imshow(flatten_layer_output.cpu())
# %%
