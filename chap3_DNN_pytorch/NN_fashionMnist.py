#%%
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.optim import SGD


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


#%%  ######## define model, loss func, optimizer

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer


#%% define function to train model

def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% func to cal accuracy of dataset
@torch.no_grad()  # no calculation of gradeint
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    # compute if loc of max in each row coincide with ground truth
    max_values, argmaxes = prediction.max(-1) # 
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


#%% train NN 
trn_dl = get_data()
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []

for epoch in range(5):
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x,y, model, optimizer, loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.array(epoch_losses).mean()
    
    # cal accuracy of prediction
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

#%% variation of training loss and accuracy over epoch
epochs = np.arange(5) + 1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()



#%% ########### scaling a dataset to improve model accuracy ######### 

#%% modify dataset to scae with 255 as max pixel in image
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 28*28)
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)
        

#%% det data loader        
def get_data():
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

#%% define model, func, opt

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer

## train batch

def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step() 
    optimizer.zero_grad()
    return batch_loss.item()

## define accuracy
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()
    
# %% train model over epochs
trn_dl = get_data()
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []

for epoch in range(5):
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        epoch_losses.append(batch_loss)
    epoch_loss = np.array(epoch_losses).mean()
    
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x,y,model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
    
#%%
epochs = np.arange(5) + 1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()

## scaling the data improved the accuracy and loss metrics





#%% ############## impact of batch size      ################

## validation dataset will be included for this analysis

# download validation dataset
val_fmnist = datasets.FashionMNIST(data_folder, train=False, download=True)

val_images = val_fmnist.data
val_targets = val_fmnist.targets


#%% define accuracy func

def accuracy(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        maxvalue, argmaxes = prediction.max(-1)
        is_correct = argmaxes == y
        return is_correct.cpu().numpy().tolist()
    
    
#%% define function to get train and val data
def get_data():
    train = FMINISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    
    val_data = FMINISTDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images))
    return trn_dl, val_dl



#%% define val loss
def val_loss(x, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        val_loss = loss_fn(prediction, y)
        return val_loss.item()
    
    
#%% fetch training and validation data
trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

#%% train model
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(5):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()
    
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    
    ##  cal val loss and accuracy
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        validation_loss = val_loss(x,y,model)
        val_epoch_accuracy = np.mean(val_is_correct)      
        
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)
    

#%% visualize results
epochs = np.arange(5)+1
import matplotlib.ticker as mticker

#%%
plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Traing loss')  
plt.plot(epochs, val_losses, 'r', label='validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss when batch size is 32')      
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.grid('off')
plt.show()



# %%
