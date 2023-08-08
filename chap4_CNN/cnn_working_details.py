
#%%
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt


#%%  ##### 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% create a toy dataset
X_train = torch.tensor([[[[1,2,3,4], 
                          [2,3,4,5],
                          [5,6,7,8],
                          [1,3,4,5]
                          ]
                         ],
                        [[[-1,2,3,-4],
                          [2,-3,4,5],
                          [-5,6,-7,8],
                          [-1,-3,-4,-5]
                          ]
                         ]
                        ]
                       ).to(device)

X_train /= 8   # normalize with max value in data to get values 0 - 1
y_train = torch.tensor([0,1]).to(device).float()










