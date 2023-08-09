
#%%
from torchvision import datasets
import torch
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#%% 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_folder = 'data/fmnist'

#%%
fmnist = datasets.FashionMNIST(data_folder, download=True,
                               train=True
                               )
tr_images = fmnist.data
tr_targets = fmnist.targets
val_fmnist = datasets.FashionMNIST(data_folder, train=False,
                                   download=True
                                   )
val_images = val_fmnist.data
val_targets = val_fmnist.targets

#%% define ata augmentation pipeline
from imgaug import augmenters as iaa
aug = iaa.Sequential(
    iaa.Affine(translate_px={'x': (-10,10)},
               mode='constant')
)

#%% define dataset class
class FMNISTDataset(Dataset):
    def __init__(self, x, y, aug=None):
        self.x, self.y = x, y
        self.aug = aug
    def __len__(self):
        return len(self.x)
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y
    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))
        if self.aug:
            ims = self.aug.augment_images(images=np.array(ims))
        ims = torch.tensor(ims)[:,None,:,:].to(device) / 255
        classes = torch.tensor(classes).to(device)
        return ims, classes


#%% define model
def get_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        nn.MaxPool2d(2), 
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=3200, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=10)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

#%%  ## define train_batch ###
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% define get data
def get_data():
    train = FMNISTDataset(x=tr_images, y=tr_targets, aug=aug)
    trn_dl = DataLoader(dataset=train, batch_size=64,
                        collate_fn=train.collate_fn, shuffle=True)
    
    val = FMNISTDataset(x=val_images, y=val_targets,
                        aug=aug)
    val_dl = DataLoader(dataset=val, batch_size=len(val_images),
                        collate_fn=val.collate_fn, shuffle=True)
    return trn_dl, val_dl



#%%
trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

#%% train over 5 epochs
for epoch in range(5):
    print(epoch)
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        
        
#%% test model on translated data
preds= []
ix = 24300
for px in range(-5, 6):
    img = tr_images[ix]/255
    img = img.view(28,28)
    img2 = np.roll(img, px, axis=1)
    plt.imshow(img2)
    plt.show()
    img3 = torch.tensor(img2).view(-1, 1, 28, 28).to(device)
    np_output = model(img3).cpu().detach().numpy()
    preds.append(np.exp(np_output)/np.sum(np.exp(np_output)))


#%% variation in predition claccs
import seaborn as sns
fig, ax = plt.subplots(1,1, figsize=(12, 10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds).reshape(11, 10),
            annot=True, ax=ax, fmt='.2f',
            xticklabels=fmnist.classes, 
            yticklabels=[str(i)+str(' pixels') for i in range(-5,6)],
            cmap='gray'
            )



# %%
