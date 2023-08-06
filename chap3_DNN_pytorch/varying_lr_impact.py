
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
    def __init__(self, x, y, scaled:bool = True):
        if scaled:
            x = x.float() / 255 
            self.x = x.view(-1, 28*28)
            
        x = x.float()
        self.x = x.view(-1, 28*28)
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    

#%% define func for loading data in batches
def get_batch_data(train_img = tr_images,
                   train_target = tr_targets,
                   val_images = val_images,
                   val_targets = val_targets,
                   batch_size: int = 32,
                   scale_data_status = True):
    train_data = FMNISTDataset(train_img, train_target, scaled=scale_data_status)
    train_dl = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True)
    
    val_data = FMNISTDataset(x=val_images, y=val_targets)
    
    # use all validation dataset
    val_dl = DataLoader(val_data, batch_size=len(val_images),
                        shuffle=False
                        )
    
    return train_dl, val_dl

# [str|callable]
#%% define model, loss function, optimizer
def get_model_specified(loss_func_type:str ='crossentropy', 
                        lr=1e-2, 
                        optim_type:str ='adam'
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
    maxvalue, argmaxes = prediction.max(-1)
    correct_pred_or_not = argmaxes == y
    return correct_pred_or_not.cpu().numpy().tolist()


#%% trigger training process func
high_lr = 1e-1
medium_lr = 1e-3
low_medium_lr = 1e-5
scaled_data = True
not_scaled_data = False


tr_dl, val_dl = get_batch_data()
model, loss_fn, optimizer = get_model_specified()
def trigger_training_process(epochs:int = 10,
                             tr_dl=tr_dl, val_dl=val_dl,
                             model=model, loss_fn=loss_fn,
                             optimizer=optimizer
                             ):
    """Returns train loss, train accuracy, validation loss,
        validation_accuracy
    """
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
        
        for ix, batch in enumerate(iter(tr_dl)):    
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
    return {'train_loss':train_loss, 
            'train_accuracy':train_accuracy, 
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
    

#%% trigger training 
#######   impact of lr on scaled dataset   ##########
## trigger training for high lr = 1e-1    

high_lr = 1e-1
medium_lr = 1e-3
low_lr = 1e-5
scaled_data = True
not_scaled_data = False

#%% scaled data with high lr


tr_dl, val_dl = get_batch_data(scale_data_status=scaled_data)    

high_model, high_loss_fn, high_optimizer = get_model_specified(lr=high_lr)

    
#%%  ## train model for scaled data with high lr
training_res= trigger_training_process(tr_dl=tr_dl, val_dl=val_dl, model=high_model, 
                                       loss_fn=high_loss_fn,
                                       optimizer=high_optimizer
                                       )   
        
#%%
scaled_high_lr_train_loss = training_res['train_loss']  
scaled_high_lr_train_acc = training_res['train_accuracy']  
scaled_high_lr_valid_loss = training_res['valid_loss']   
scaled_high_lr_valid_acc = training_res['valid_accuracy']


#%%
plot_loss(train_loss=scaled_high_lr_train_loss,
          valid_loss=scaled_high_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for scaled data with high lr ={high_lr}'
          )

#%%
plot_accuracy(train_accuracy=scaled_high_lr_train_acc,
          valid_accuracy=scaled_high_lr_valid_acc,
          num_epochs=10, 
          title=f'Traing and validation Accuracy for scaled data with high lr ={high_lr}'
          )




#%%   #####  trained model on scaled data with medium lr    ######

tr_dl, val_dl = get_batch_data(scale_data_status=scaled_data)
med_model, med_loss_fn, med_optimizer = get_model_specified(lr=medium_lr)

#%% train medium lr model
scaled_med_lr_training_res = trigger_training_process(tr_dl=tr_dl, val_dl=val_dl, 
                                                    model=med_model, loss_fn=med_loss_fn,
                                                    optimizer=med_optimizer
                                                    )

#%%  
scaled_med_lr_train_loss = scaled_med_lr_training_res['train_loss']  
scaled_med_lr_train_acc = scaled_med_lr_training_res['train_accuracy']  
scaled_med_lr_valid_loss = scaled_med_lr_training_res['valid_loss']   
scaled_med_lr_valid_acc = scaled_med_lr_training_res['valid_accuracy']

#%%
plot_loss(train_loss=scaled_med_lr_train_loss,
          valid_loss=scaled_med_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for scaled data with medium lr ={medium_lr}'
          )

#%%
plot_accuracy(train_accuracy=scaled_med_lr_train_acc,
          valid_accuracy=scaled_med_lr_valid_acc,
          num_epochs=10, 
          title=f'Traing and validation Accuracy for scaled data with medium lr ={medium_lr}'
          )




#%%

#%%   #####  trained model on scaled data with low lr    ######

tr_dl, val_dl = get_batch_data(scale_data_status=scaled_data)
low_model, low_loss_fn, low_optimizer = get_model_specified(lr=low_lr)

#%% train medium lr model
scaled_low_lr_training_res = trigger_training_process(tr_dl=tr_dl, val_dl=val_dl, 
                                                    model=low_model, loss_fn=low_loss_fn,
                                                    optimizer=low_optimizer,
                                                    epochs=100
                                                    )

#%%  
scaled_low_lr_train_loss = scaled_low_lr_training_res['train_loss']  
scaled_low_lr_train_acc = scaled_low_lr_training_res['train_accuracy']  
scaled_low_lr_valid_loss = scaled_low_lr_training_res['valid_loss']   
scaled_low_lr_valid_acc = scaled_low_lr_training_res['valid_accuracy']

#%%
plot_loss(train_loss=scaled_low_lr_train_loss,
          valid_loss=scaled_low_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for scaled data with low lr ={low_lr}'
          )

#%%
plot_accuracy(train_accuracy=scaled_low_lr_train_acc,
          valid_accuracy=scaled_low_lr_valid_acc,
          num_epochs=10, 
          title=f'Traing and validation Accuracy for scaled data with low lr ={low_lr}'
          )



#%%  ### parameter distribution across layers for different lr

def plot_model_parameter_dist(model):
    for ix, par in enumerate(model.parameters()):
        if(ix==0):
            plt.hist(par.cpu().detach().numpy().flatten())
            plt.title('Distribution of weights connecting inputs to hidden layer')
            plt.show()
        elif(ix==1):
            plt.hist(par.cpu().detach().numpy().flatten())
            plt.title('Distribution of biases of hidden layers')
            plt.show()
        elif(ix==2):
            plt.hist(par.cpu().detach().numpy().flatten())
            plt.title('Distribution of weights connecting hidden to output layers')
            plt.show()
        elif(ix==3):
            plt.hist(par.cpu().detach().numpy().flatten())
            plt.title('Distribution of biases of output layer')
            plt.show()
            
            
            
#%%
plot_model_parameter_dist(model=low_model)

#%%
plot_model_parameter_dist(model=med_model)

#%%
plot_model_parameter_dist(model=high_model)


#%%  ############# no scaling of data with various learning rate  -- high lr   ##########

noscaled_tr_dl, noscaled_val_dl = get_batch_data(scale_data_status=not_scaled_data)
#get_model_specified(lr=high_lr)

#%%
noscaled_high_lr_train_res = trigger_training_process(tr_dl=noscaled_tr_dl, 
                                                      val_dl=noscaled_val_dl,
                                                    model=high_model, 
                                                    loss_fn=high_loss_fn, 
                                                    optimizer=high_optimizer
                                                    )


#%%  ########      ########

noscaled_high_lr_train_loss = noscaled_high_lr_train_res['train_loss']  
noscaled_high_lr_train_acc = noscaled_high_lr_train_res['train_accuracy']  
noscaled_high_lr_valid_loss = noscaled_high_lr_train_res['valid_loss']   
noscaled_high_lr_valid_acc = noscaled_high_lr_train_res['valid_accuracy']

#%%
plot_loss(train_loss=noscaled_high_lr_train_loss,
          valid_loss=noscaled_high_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for not scaled data with high lr ={high_lr}'
          )

#%%
plot_accuracy(train_accuracy=noscaled_high_lr_train_acc,
             valid_accuracy=noscaled_high_lr_valid_acc,
              num_epochs=10, 
              title=f'Traing and validation Accuracy for not scaled data with high lr ={high_lr}'
          )



#%%   #####  no scaled data medium lr  #####
noscaled_med_lr_train_res = trigger_training_process(tr_dl=noscaled_tr_dl, 
                                                      val_dl=noscaled_val_dl,
                                                        model=med_model, 
                                                        loss_fn=med_loss_fn, 
                                                        optimizer=med_optimizer
                                                    )


#%%  ########      ########

noscaled_med_lr_train_loss = noscaled_med_lr_train_res['train_loss']  
noscaled_med_lr_train_acc = noscaled_med_lr_train_res['train_accuracy']  
noscaled_med_lr_valid_loss = noscaled_med_lr_train_res['valid_loss']   
noscaled_med_lr_valid_acc = noscaled_med_lr_train_res['valid_accuracy']

#%%
plot_loss(train_loss=noscaled_med_lr_train_loss,
          valid_loss=noscaled_med_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for not scaled data with medium lr ={medium_lr}'
          )

#%%
plot_accuracy(train_accuracy=noscaled_med_lr_train_acc,
             valid_accuracy=noscaled_med_lr_valid_acc,
              num_epochs=10, 
              title=f'Traing and validation Accuracy for not scaled data with medium lr ={medium_lr}'
          )




#%%   #####  no scaled data low lr  #####
noscaled_low_lr_train_res = trigger_training_process(tr_dl=noscaled_tr_dl, 
                                                      val_dl=noscaled_val_dl,
                                                        model=low_model, 
                                                        loss_fn=low_loss_fn, 
                                                        optimizer=low_optimizer
                                                    )


#%%  ########      ########

noscaled_low_lr_train_loss = noscaled_low_lr_train_res['train_loss']  
noscaled_low_lr_train_acc = noscaled_low_lr_train_res['train_accuracy']  
noscaled_low_lr_valid_loss = noscaled_low_lr_train_res['valid_loss']   
noscaled_low_lr_valid_acc = noscaled_low_lr_train_res['valid_accuracy']

#%%
plot_loss(train_loss=noscaled_low_lr_train_loss,
          valid_loss=noscaled_low_lr_valid_loss,
          num_epochs=10, 
          title=f'Traing and validation loss for not scaled data with low lr ={low_lr}'
          )

#%%
plot_accuracy(train_accuracy=noscaled_low_lr_train_acc,
             valid_accuracy=noscaled_low_lr_valid_acc,
              num_epochs=10, 
              title=f'Traing and validation Accuracy for not scaled data with low lr ={medium_lr}'
          )

#%%

print('=====  Parameter distribution in layers of models =====')
print(f'==== Not scaled low lr = {low_lr}')
plot_model_parameter_dist(model=low_model)

print('=====  Parameter distribution in layers of models =====')
print(f'==== Not scaled medium lr = {medium_lr}')
plot_model_parameter_dist(model=med_model)

print('=====  Parameter distribution in layers of models =====')
print(f'==== Not scaled high lr = {high_lr}')
plot_model_parameter_dist(model=high_model)






 
#%%  #######  learning rate annealing  #########
from torch import optim

#%%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                 patience=0, threshold = 0.001,
                                                 verbose=True, min_lr=1e-5,
                                                 threshold_mode='abs')

def trigger_training_with_lr_annealing(epochs, tr_dl, val_dl, model, loss_fn,
                                       optimizer):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(epochs):
        print(epoch)
        train_epoch_losses, train_epoch_accuracies = [], []
        for ix, batch in enumerate(iter(tr_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        
        for ix, batch in enumerate(iter(tr_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy =np.mean(train_epoch_accuracies)
        
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model)
            validation_loss = val_loss(x, y, model)
            scheduler.step(validation_loss)
        val_epoch_accuracy = np.mean(val_is_correct)
        
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

    return {'train_loss':train_losses, 
            'train_accuracy':train_accuracies, 
            'valid_loss': val_losses, 
            'valid_accuracy':val_accuracies
         }
        

#%%

tr_dl, val_dl = get_batch_data()
lr_anneal_model, loss_fn, optimizer = get_model_specified(lr=1e-3)

# %%
lr_anneal_train_res = trigger_training_with_lr_annealing(epochs=100, tr_dl=tr_dl,
                                                         val_dl=val_dl, model=lr_anneal_model, 
                                                         loss_fn=loss_fn,
                                                         optimizer=optimizer
                                                         )
# %%






