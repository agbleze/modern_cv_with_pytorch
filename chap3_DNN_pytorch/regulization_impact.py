
#%% import modules
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker



#%%
data_folder = 'data/fmnist'
fmnist = datasets.FashionMNIST(data_folder, train=True, download=True)
tr_images, tr_targets = fmnist.data, fmnist.targets

fmnist_val = datasets.FashionMNIST(data_folder, train=False, download=True)
val_images, val_targets = fmnist_val.data, fmnist_val.targets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%% define data

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 28*28)
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    
    
#%% define get data
def get_data(batch_size=32):
    tr_data = FMNISTDataset(tr_images, tr_targets)
    tr_dl = DataLoader(tr_data, batch_size=batch_size)
    
    val_data = FMNISTDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images))
    return tr_dl, val_dl


#%% define model, loss-fn, optimizer

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


#%% define val loss function
@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()

#%% define accuracy function
@torch.no_grad()
def accuracy(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    max_value, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


#%%  ### defin training function with l1 regularization
def train_batch_with_regularization(model, loss_fn, optimizer,
                                    regularization_type = 'l1',
                                    x=None, y=None
                                    ):
    model.train()
    prediction = model(x)
    regularization = 0
    
    if regularization_type == 'l1':
        for param in model.parameters():
            regularization += torch.norm(param, 1)
        batch_loss = loss_fn(prediction, y) + 0.0001 * regularization
    else:
        for param in model.parameters():
            regularization += torch.norm(param, 2)
        batch_loss = loss_fn(prediction, y) + 0.01 * regularization
        
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%% define trigger training procedure

def trigger_train_process(num_epochs, train_dataload, val_dataload,
                          model, loss_fn, optimizer,
                          train_batch_fn: callable,
                          regularization_type: str
                          ):
    train_loss, train_accuracies = [], []
    valid_loss, valid_accuracy = [], []
    
    for epoch in range(num_epochs):
        print(epoch)
        train_epoch_loss, train_epoch_accuracies = [], []
        
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            batch_loss = train_batch_fn(x=x, y=y,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        regularization_type=regularization_type
                                        )
            train_epoch_loss.append(batch_loss)
            
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            is_correct = accuracy(x, y, model=model, loss_fn=loss_fn)
            train_epoch_accuracies.extend(is_correct)
            
            
        for ix, batch in enumerate(iter(val_dataload)):
            x, y = batch
            valid_losses = val_loss(x, y, model, loss_fn)
            is_correct = accuracy(x, y, model, loss_fn)
            
        train_loss.append(np.array(train_epoch_loss).mean())
        train_accuracies.append(np.array(train_epoch_accuracies).mean())
        valid_loss.append(valid_losses)
        valid_accuracy.append(np.array(is_correct).mean())
    return {'train_loss':train_loss, 
            'train_accuracy':train_accuracies, 
            'valid_loss': valid_loss, 
            'valid_accuracy':valid_accuracy
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
    
    
#%%

tr_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()
#l1_train_batch_fn = train_batch_with_regularization()

#%%

l1_train_res = trigger_train_process(num_epochs=10, train_dataload=tr_dl,
                                     val_dataload=val_dl, model=model, 
                                     loss_fn=loss_fn,
                                     optimizer=optimizer, 
                                     train_batch_fn=train_batch_with_regularization,
                                     regularization_type='l1'
                                     )

#%%
l1_train_res.keys()
        
#%%
l1_train_loss = l1_train_res['train_loss']
l1_val_loss = l1_train_res['valid_loss']
l1_train_acc = l1_train_res['train_accuracy']
l1_val_acc = l1_train_res['valid_accuracy']

#%%   #######      ########
plot_loss(train_loss=l1_train_loss, valid_loss=l1_val_loss, 
          num_epochs=10,
          title='Training and validation loss with very small input values and L1 regularization'
          )

plot_accuracy(train_accuracy=l1_train_acc, valid_accuracy=l1_val_acc,
              num_epochs=10,
              title='Training and validation accuracy with very small input values and L1 regularization'
              )


#%%  ##### L2 regularization #####

l2_train_res = trigger_train_process(num_epochs=10, train_dataload=tr_dl,
                                     val_dataload=val_dl, model=model, 
                                     loss_fn=loss_fn,
                                     optimizer=optimizer, 
                                     train_batch_fn=train_batch_with_regularization,
                                     regularization_type='l2'
                                     )


#%%
l2_train_loss = l2_train_res['train_loss']
l2_val_loss = l2_train_res['valid_loss']
l2_train_acc = l2_train_res['train_accuracy']
l2_val_acc = l2_train_res['valid_accuracy']


#%%   #######      ########
plot_loss(train_loss=l2_train_loss, valid_loss=l2_val_loss, 
          num_epochs=10,
          title='Training and validation loss with very small input values and L2 regularization'
          )

plot_accuracy(train_accuracy=l2_train_acc, valid_accuracy=l2_val_acc,
              num_epochs=10,
              title='Training and validation accuracy with very small input values and L2 regularization'
              )





#%%  ########  IMAGE TRANSLATION  ########
import matplotlib.pyplot as plt

#%%
ix = np.random.randint(len(tr_images))

ix = 24300
plt.imshow(tr_images[ix], cmap='gray')
plt.title(fmnist.classes[tr_targets[ix]])


# %%
tr_targets[ix]
# %%
fmnist.classes
# %% ##### Pass the image through the trained model  ######
# preprocess the image 
img = tr_images[ix] / 255
img = img.view(28*28)
img = img.to(device)

#%%  #### extract the probabilities of various classes
np_output = model(img).cpu().detach().numpy()
np.exp(np_output) / np.sum(np.exp(np_output))

#%%
np_output

#%% translate (roll / slide) the image multiple times 5 pixels to left and right
# create a list that stores predictions
preds = []

for px in range(-5, 6):
    # preprocess the img
    img = tr_images[ix] / 255
    img = img.view(28, 28)
    
    # roll the img by px 
    img2 = np.roll(img, px, axis=1)
    
    # store rolled img and register it to device
    img3 = torch.Tensor(img2).view(28*28).to(device)
    
    # predict class of image
    np_output = model(img3).cpu().detach().numpy()
    preds.append(np.exp(np_output)/np.sum(np.exp(np_output)))
    
    
#%% visualize the predictions of the model
import seaborn as sns
fig, ax = plt.subplots(1,1, figsize=(12,10))

plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds), annot=True, ax=ax, fmt='.2f',
            xticklabels=fmnist.classes, yticklabels=[str(i)+str('pixels') for i in range(-5,6)],
            cmap='gray'
            )





# %%
