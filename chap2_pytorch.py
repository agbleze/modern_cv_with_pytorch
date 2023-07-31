
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

x.to(device)
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

            




