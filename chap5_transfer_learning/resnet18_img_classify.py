
#%%
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import cv2
import matplotlib.ticker as mticker



#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

train_data_dir = 'training_set'
test_data_dir = 'test_set'


#%% ### build residual block architecture ###

class ResLayer(nn.Module):
    def __init__(self, ni, no, kernel_size, stride=1):
        super(ResLayer, self).__init__()
        padding = kernel_size -2
        self.conv = nn.Sequential(
            nn.Conv2d(ni, no, kernel_size, stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x) +x
        return x



#%% ### resNet block architecture  -- resnet18 for pretraining

## define class for creating data point

class CatsDogs(Dataset):
    def __init__(self, folder, num_images):
        cats = glob(folder+'/cats/*.jpg')
        dogs = glob(folder+'/dogs/*.jpg')
        if not num_images:
            self.fpaths = cats + dogs
        else:
            self.fpaths = cats[:num_images] + dogs[:num_images]
            
        from random import seed, shuffle
        seed(10); shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog')
                        for fpath in self.fpaths
                        ]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                              )
        
            
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224, 224))
        im = torch.tensor(im/255)
        im = im.permute(2,0,1)
        im = self.normalize(im)
        return (im.float().to(device),
                torch.tensor([target]).float().to(device)
                )
        
        
        
#%% define func for fetching datasets in batches
def get_data(batch_size=32, num_images=500):
    train = CatsDogs(train_data_dir, num_images=num_images)
    trn_dl = DataLoader(dataset=train, batch_size=batch_size, 
                        shuffle=True, drop_last=True
                        )
    
    val = CatsDogs(test_data_dir, num_images=num_images)
    val_dl = DataLoader(dataset=val, batch_size=batch_size, 
                        shuffle=True, drop_last=True
                        )
    return trn_dl, val_dl

#%% define model architecture, loss_fn, optimizer
# use pretrained resNet18
resnet18 = models.resnet18(pretrained=True)
def get_model(pretrained_model=resnet18):
    model = pretrained_model
    # frezee model parameters to prevent training and weight updates
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Flatten(),
                             nn.Linear(512, 128),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(128, 1),
                             nn.Sigmoid()
                             )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer

#%% define train batch
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
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()


#%% ### define training process  #####
trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model(pretrained_model=resnet18)

#%% define process of training ###
def trigger_training_process(train_dataload, val_dataload, model, loss_fn, optimizer,
                             num_epochs=5
                             ):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []
        
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
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
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        
        train_losses.append(train_epoch_accuracy)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
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
    


#%%  ###  ######
restnet18_train_res = trigger_training_process(trn_dl, val_dl, model, loss_fn, 
                                               optimizer)




#%% ###### train and visualize acc for resnet34,50,101,152 on 1k data points
