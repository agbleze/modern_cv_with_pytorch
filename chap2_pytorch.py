
#%%
import torch

#%%
x = torch.tensor([[2., -1.],[1., 1.]], requires_grad=True)
print(x)

out = x.pow(2).sum()

#%% cal gradient
out.backward()
x.grad

# %% building NN with pytorch on toy dataset
x = [[1,2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

x = torch.tensor(x).float()
Y = torch.tensor(y).float()

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = x.to(device)
Y = Y.to(device)

#%%

import torch.nn as nn

#%%

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)
        
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
    
# feedforward propagation always has to be defined as forwar func
#%%
torch.manual_seed(2023)
mynet = MyNeuralNet().to(device)

#%% checking weight and bias of layer
mynet.input_to_hidden_layer.weight

#%% get all parameters of the nn
mynet.parameters()

# obtain parameters by looping through

for par in mynet.parameters():
    print(par)

#%% define loss function
loss_func =  nn.MSELoss()           

#%% cal loss for input
_Y = mynet(X)

loss_value = loss_func(_Y, Y)
print(loss_value)

#%% optimizer

from torch.optim import SGD

#%% optimizer

opt = SGD(mynet.parameters(), lr=0.001)

loss_history = []

for i in range(50):   
    loss_value = loss_func(mynet(X), Y) # compute loss
    loss_value.backward() # perform backward propagation
    opt.step() # update weight according to gradient comuted
    loss_history.append(loss_value)
    
    
#%% plot loss over epochs

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')


#%% Dataset, dataloader, batch size
# 1. import methods for loading data and dealing with datasets

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

#%%  2. import data, convert to float and register on device
x = [[1,2], [3,4], [5,6], [7,8]]
y = [[3], [7], [11], [15]]

# convert to float
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# register data to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = X.to(device)
Y = Y.to(device)

#%% 3. define dataset class

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]
    
#%% 4. create instance of class
ds = MyDataset(X, Y)
        
#%% 5. pass dataset instance into Dataloader to fetch batch_size
dl = DataLoader(dataset=ds, batch_size=2, shuffle=True)

for x,y in dl:
    print(x,y)

#%% 6. define the neural network class
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.x = torch.tensor(x).float()
        #self.y = torch.tensor(y).float()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)
        #self.hidden_activation = nn.ReLU()
        #self.output_layer = nn.Linear(1)
        
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
        
#%% 7. initialize model, loss functio, optimizer

mynet = MyNeuralNet()
loss_func = nn.MSELoss()

opt = SGD(lr=0.001)

#%% 8. loop through batches of data and minimize loss
import time

loss_history = []

start = time.time()

for _ in range(50):
    for data in dl:
        x, y = data
        opt.zero_grad()
        loss_value = loss_func(mynet(x), y)
        loss_value.backward()
        opt.step()
    loss_history.append(loss_value)
    end = time.time()
    print(end-start)
    

#%%  ###### Predicting on new datapoints #####
# 1. create data points
val_x = [[10, 11]]
# 2. convert data point to tensor and register on device
val_x = torch.tensor(val_x).float().to(device)

# 3. pass tensor object through traine NN to obtain prediction
mynet(val_x)




