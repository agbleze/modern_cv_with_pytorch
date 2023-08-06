#%%
import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam


#%% ### GET DATASET ###
data_folder = 'data/fmnist'
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
        x = x.float() / 255
        self.x = x.view(-1, 28*28)
        self.y = y
        #self.y = y.float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        x = self.x[ix]
        y = self.y[ix]
        return x.to(device), y.to(device)
    
    
#%% define class for loading batches of data

def get_data():
    tr_data = FashionDataset(tr_images, tr_targets)
    tr_dl = DataLoader(tr_data, batch_size=32, shuffle=True)
    
    val_data = FashionDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images), shuffle=False)
    return tr_dl, val_dl


#%% define model, loss_fn, opt
def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    
    optimizer = SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, optimizer


#%% define traing batch fun
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

#%% define func for accuracy
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_value, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()
    

#%% defin val loss
@torch.no_grad()
def val_loss(x,y, model, loss_fn):
    model.eval()
    val_prediction = model(x)
    val_loss = loss_fn(val_prediction, y)
    return val_loss.to(device).tolist()



#%% implement training batches of data per epoch

# get batch of data and model
tr_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

#%% setup train results to be monitored
training_loss, valid_loss = [], []
training_accuracy, valid_accuracy = [], []

#%% loop through epoch and train the model

for epoch in range(10):
    print(epoch)
    training_epoch_loss, training_epoch_accuracy = [], []
    for ix, batch in enumerate(iter(tr_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        training_epoch_loss.append(batch_loss)
        
        # cal accuracy for training data
    for ix, batch in enumerate(iter(tr_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        training_epoch_accuracy.extend(is_correct)
    
    training_loss.append(np.mean(np.array(training_epoch_loss)))
    training_accuracy.append(np.mean(np.array(training_epoch_accuracy)))
    
    
    # compute val 
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        valid_losses = val_loss(x, y, model, loss_fn)
        
        valid_accuracies = accuracy(x, y, model)
    valid_loss.append(valid_losses)
    valid_accuracy.append(np.mean(valid_accuracies))
    
    
#%% #### visualize loss and accuracy ####
import matplotlib.ticker as mticker
epochs = np.arange(10)+1
plt.subplot(111)
plt.plot(epochs, training_loss, 'bo', label='Training loss')
plt.plot(epochs, valid_loss, 'r', label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')   
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss with SGD optimizer and batch size=32') 
plt.legend()
plt.grid('off')
plt.show() 

#%% 
plt.subplot(121)
plt.plot(epochs, training_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, valid_accuracy, 'g', label='validation accuracy')
plt.xlabel('Epoch') 
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.title('Training and validation accuracy with SGD optimizer - batch_size=32')
plt.grid('off')
plt.legend()
plt.show()


    
#%% ##### change optimizer to Adam ######

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer


#%% ####  train model  #####
train_loss, valid_loss = [], []
train_accuracy, valid_accuracy = [], []
model, loss_fn, optimizer = get_model()

#%%

for epoch in range(10):
    print(epoch)
    train_epoch_loss, train_epoch_accuracy = [], []  
    for ix, batch in enumerate(iter(tr_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)
        train_epoch_loss.append(batch_loss)
        
    for ix, batch in enumerate(iter(tr_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracy.extend(is_correct)
    train_loss.append(np.mean(np.array(train_epoch_loss)))
    mean_train_acc = np.mean(np.array(train_epoch_accuracy))
    train_accuracy.append(mean_train_acc)
    
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        valid_losses = val_loss(x, y, model, loss_fn)
        valid_is_correct = accuracy(x, y, model)
    valid_loss.append(valid_losses)
    valid_accuracy.append(np.mean(np.array(valid_is_correct)))
    
#%%
epochs = np.arange(10)+1
plt.subplot(111)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, valid_loss, 'r', label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & validation loss with Adam optimizer - batch size = 32')
plt.grid('off')
plt.legend()
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.show()

#%%
plt.subplot(121)
plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, valid_accuracy, 'r', label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & validation Accuracy with Adam optimizer - batch size = 32')
plt.grid('off')
plt.legend()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.show()


        
        






# %%
