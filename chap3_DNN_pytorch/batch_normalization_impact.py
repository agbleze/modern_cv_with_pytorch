
#%%
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset 
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
import matplotlib.ticker as mticker


#%%  #####  Very small input values with batch normalization  #####




#%%  #### get data #####
data_folder = 'data/fmnist'

fmnist = datasets.FashionMNIST(data_folder, train=True, download=True)
tr_images = fmnist.data 
tr_targets = fmnist.targets

fmnist_val = datasets.FashionMNIST(data_folder, train=False, download=True)
val_images = fmnist_val.data
val_targets = fmnist_val.targets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%% ##### create dataset normalize to have very small values
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / (255*10_000)
        x = x.view(-1, 28*28)
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

def get_data():
    tr_data = FMNISTDataset(tr_images, tr_targets)
    tr_dl = DataLoader(tr_data, batch_size=32, shuffle=True)
    
    val_data = FMNISTDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images), shuffle=False)
    return tr_dl, val_dl

    
#%% define NN model architecture
def get_model():
    class NeuralNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_to_hidden_layer = nn.Linear(28*28, 1000)
            self.hidden_layer_activation = nn.ReLU()
            self.hidden_to_output_layer = nn.Linear(1000, 10)
            
        def forward(self,x):
            x = self.input_to_hidden_layer(x)
            x0 = self.hidden_layer_activation(x)
            x1 = self.hidden_to_output_layer(x0)
            return x1, x0
    model = NeuralNet()
    loss_fn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer




#%% 

def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)[0]
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% define accuracy fn
torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)[0]
    maxvalue, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


#%% define val loss function
torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    pred = model(x)[0]
    val_losses = loss_fn(pred, y)
    return val_losses.item()


#%%

def trigger_train_process(tr_dl, val_dl, model, loss_fn, optimizer, epochs=10):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        print(epoch)
        train_epoch_loss, train_epoch_accuracies = [], []
        #val_epoch_loss, val_epoch_accuracies = [], []
        
        for ix, batch in enumerate(iter(tr_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
            train_epoch_loss.append(batch_loss)
        
        for ix, batch in enumerate(iter(tr_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
            
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            valid_epoch_loss = val_loss(x, y, model, loss_fn)
            valid_epoch_accuracies = accuracy(x, y, model)
            
        train_losses.append(np.array(train_epoch_loss).mean())   
        train_epoch_accuracy = np.array(train_epoch_accuracies).mean()
        train_accuracies.append(train_epoch_accuracy)
        
        val_losses.append(valid_epoch_loss)
        val_accuracies.append(np.array(valid_epoch_accuracies).mean())
        
    return {'train_losses': train_losses,
            'train_accuracy': train_accuracies,
            'valid_loss': val_losses,
            'valid_accuracy': val_accuracies
            }
       

#%%  ####   #####

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
    
#%%  ######     #######

tr_dl, val_dl = get_data()

model, loss_fn, optimizer = get_model()

#%%
small_value_train_res = trigger_train_process(tr_dl=tr_dl, val_dl=val_dl, 
                                                model=model, loss_fn=loss_fn,
                                                optimizer=optimizer
                                                )
        

#%%
small_value_train_res.keys()

#%%  ##### plots  #####
smvl_train_loss = small_value_train_res['train_losses']
smvl_val_loss = small_value_train_res['valid_loss']
smvl_train_acc = small_value_train_res['train_accuracy']
smvl_val_acc = small_value_train_res['valid_accuracy']

#%%

plot_loss(smvl_train_loss, valid_loss=smvl_val_loss, num_epochs=10,
          title='Training and validation loss with small input values'
          )
        
#%% 
plot_accuracy(train_accuracy=smvl_train_acc, valid_accuracy=smvl_val_acc, num_epochs=10,
              title='Training and validation accuracy with small input values'
              )
            

#%%  #####  batch normalization  #####

def get_batch_normalized_model():
    class neuralnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_to_hidden_layer = nn.Linear(28*28, 1000)
            self.batch_norm = nn.BatchNorm1d(1000)
            self.hidden_layer_activation = nn.ReLU()
            self.hidden_to_output_layer = nn.Linear(1000, 10)
            
        def forward(self, x):
            x = self.input_to_hidden_layer(x)
            x0 = self.batch_norm(x)
            x1 = self.hidden_layer_activation(x0)
            x2 = self.hidden_to_output_layer(x1)
            return x2, x1
            
    model = neuralnet().to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer







# %%
