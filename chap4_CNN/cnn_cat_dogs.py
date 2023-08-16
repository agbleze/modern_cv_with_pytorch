#%%
import torchvision
import torch.nn as nn
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.ticker as mticker

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
train_data_dir = 'archive/training_set/training_set'
test_data_dir = 'archive/test_set/test_set'

#%% move filenames begining with dog to dogs folder
#  !mv dogs-vs-cats/train/dog* dogs-vs-cats/train/dogs;

#%%
#glob(train_data_dir+'/*cat*.jpg')

#%%
#glob(train_data_dir+'/*dog*.jpg')

#%%
class cats_dogs(Dataset):
    def __init__(self, folder, num_images = None):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*jpg')
        if num_images is None:
            self.fpaths = cats + dogs
        else:
            self.fpaths = cats[:num_images] + dogs[:num_images]
        
        from random import shuffle, seed
        seed(10)
        shuffle(self.fpaths)
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


#%%

cats_dogs(test_data_dir)
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
def get_data(num_images=None):
    train = cats_dogs(train_data_dir, num_images=num_images)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
    val = cats_dogs(test_data_dir, num_images=num_images)
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

def trigger_train_process(train_dataload, val_dataload, 
                          model, loss_fn, optimizer,
                          num_epochs: int = 5,
                          train_batch = train_batch):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(num_epochs):
        print(epoch)
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []
        
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            batch_loss = train_batch(x, y, model, 
                                     loss_fn, optimizer
                                     )
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        
        for ix, batch in enumerate(iter(val_dataload)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
            validation_loss = val_loss(x, y, model, loss_fn)
        val_epoch_accuracy = np.mean(train_epoch_accuracies)
        
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        val_losses.append(validation_loss)
        
    return {'train_loss':train_losses, 
            'train_accuracy':train_accuracies, 
            'valid_loss': val_losses, 
            'valid_accuracy':val_accuracies
        }
        
        
#%%
def plot_loss(train_loss, valid_loss, num_epochs=10, 
              title='Training and validation loss lr - 0.01'
              ):
    epochs = np.arange(num_epochs)+1
    plt.subplot(111)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss, 'r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.show()
 
#%% 
def plot_accuracy(train_accuracy, valid_accuracy, num_epochs, title):
    epochs = np.arange(num_epochs)+1
    plt.subplot(111)
    plt.plot(epochs, train_accuracy, 'bo', label='Training Accuracy')
    plt.plot(epochs, valid_accuracy, 'r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.show()  
    
#%%  ######      
model, loss_fn, optimizer = get_model()

#%%
train_dataload, val_dataload = get_data()    


#%%
all_img_train_res = trigger_train_process(train_dataload=train_dataload,
                                            val_dataload=val_dataload,
                                            model=model,loss_fn=loss_fn, 
                                            optimizer=optimizer
                                            )


#%%   #####   ######


    
#%%    
#! kaggle datasets download -d tongpython/cat-and-dog    
    
    
    
# %%
