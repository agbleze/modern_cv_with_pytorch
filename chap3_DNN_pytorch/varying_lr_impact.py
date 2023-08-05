
#%%
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss


#%% get data for modeling
data_folder = 'data/fmnist'

fmnist = datasets.FashionMNIST(data_folder, train=True, download=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

# download validation images
fmnist_val = datasets.FashionMNIST(data_folder, train=False, download=True)
val_images = fmnist_val.data
val_targets = fmnist_val.targets

#%% define class for loading datasets
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class FMNISTDataset(Dataset):
    def __init__(self, x, y, scaled: bool):
        if scaled:
            self.x = x.float() / 255 
        self.x = x.float()
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    

#%% define func for loading data in batches
def get_batch_data(train_img = tr_images,
                   train_target = tr_targets,
                   val_images = val_images,
                   val_targets = val_images,
                   batch_size: int = 32):
    train_data = FMNISTDataset(train_img, train_target)
    train_dl = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True)
    
    val_data = FMNISTDataset(x=val_images, y=val_targets)
    
    # use all validation dataset
    val_dl = DataLoader(val_data, batch_size=len(val_images),
                        shuffle=False
                        )
    
    return train_dl, val_dl


#%% define model, loss function, optimizer
def get_model_specified(loss_func_type: [str|callable]='crossentropy', 
                        lr=1e-2, 
                        optim_type: [str|callable]='adam'
                        ):
    """Returns model, loss function, and optimizer"""
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10) # output predicts 10 classes
    ).to(device)
    
    if loss_func_type == 'crossentropy':
        loss_fn = CrossEntropyLoss()
    else:
        loss_fn = loss_func_type()
        
    if optim_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optim_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim_type()
        
    return model, loss_fn, optimizer
    
#%% validation loss function
@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_losses = loss_fn(prediction, y)
    return val_losses.item()


#%% define func for training data
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% define func for cal accuracy
@torch.no_grad()
def accuracy(x, y, model):
    prediction = model(x)
    correct_pred_or_not = prediction.max(-1)
    maxvalue, argmaxes = correct_pred_or_not == y
    return argmaxes.cpu().numpy().tolist()


#%% trigger training process func

def trigger_training_process(epochs: int = 10):
    """Returns train loss, train accuracy, validation loss,
        validation_accuracy
    """
    tr_dl, val_dl = get_batch_data()
    model, loss_fn, optimizer = get_model_specified()
    train_loss, train_accuracy = [], []
    valid_loss, valid_accuracy = [], []
    
    
    # loop over train batch data and train model
    # for each epoch
    for epoch in range(epochs):
        print(epoch)
        # store loss and acc for each batch during epoch
        tr_epoch_losses, tr_epoch_acc = [], []
        for ix, batch in enumerate(iter(tr_dl)):
            x, y = batch  # load features and labels for batch
            batch_loss = train_batch(x, y, model, loss_fn,
                                     optimizer
                                     )
            tr_epoch_losses.append(batch_loss)
            
            tr_acc = accuracy(x, y, model)
            tr_epoch_acc.extend(tr_acc)
            # after batch finishes cal mean loss & acc 
            # of all batch data for that epoch
        train_loss.append(np.mean(tr_epoch_losses))
        mean_ep_acc = np.mean(np.array(tr_epoch_acc))
        train_accuracy.append(mean_ep_acc)
        
        # cal loss and acc for validation dataset per epoch
        # all validation dataset are used per batch for evaluation
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_losses = val_loss(x, y, model, loss_fn)
            
            # returns results for where class predicted is correct
            # for 10 output classes
            is_correct_pred = accuracy(x, y, model)
        valid_loss.append(val_losses)
        valid_accuracy.append(np.mean(np.array(is_correct_pred)))
    return train_loss, train_accuracy. valid_loss, valid_accuracy
        
        
    
        
       
        






