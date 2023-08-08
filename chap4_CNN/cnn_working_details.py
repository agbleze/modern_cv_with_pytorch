
#%%
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.optim import SGD, Adam
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary


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
                       ).to(device).float()

X_train /= 8   # normalize with max value in data to get values 0 - 1
y_train = torch.tensor([0,1]).to(device).float()

#%% define model architecture
def get_model():
    model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1, 1),
                nn.Sigmoid()
            ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


#%%  summarize model architecture
model, loss_fn, optimizer = get_model()

summary(model=model, input_data=X_train)

#%%  #### define func to train on batch of data
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction.squeeze(0), y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


#%%  #### define train dataloader
trn_dl = DataLoader(TensorDataset(X_train, y_train))


#%% train the model
for epoch in range(2000):
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, loss_fn, optimizer)

#%% forward pass on 1st datapoint
model(X_train[:1])

#%%  ######## How CNN works for forward propagation  #####
# #### extract the various layers of the model  ###
list(model.children())


#%% ## extarct layers with weight

(cnn_w, cnn_b), (lin_w, lin_b) = [(layer.weight.data, layer.bias.data)
                                  for layer in list(model.children())
                                  if hasattr(layer, 'weight')
                                  ]

# %% perform convolution operation over input value
## get image height and width
## make imput matrix filled with zeros
h_im, w_im = X_train.shape[2:]
h_conv, w_conv = cnn_w.shape[2:]
sumprod = torch.zeros((h_im - h_conv + 1, w_im - w_conv +1))

#%% fill sumprod by convolving the filter (cnn_w) 
# across the 1st input and sum up the filter bias term (cnn_b)

for i in range(h_im - h_conv + 1):
    for j in range(w_im - w_conv +1):
        # take portion of image that needs to be convolved over with filter
        img_subset = X_train[0, 0, i:(i+3), j:(j+3)]
        model_filter = cnn_w.reshape(3,3)
        val = torch.sum(img_subset*model_filter) + cnn_b
        sumprod[i, j] = val


#%% ReLU activation is applied to the convolved output
sumprod.clamp_min(0)

#%% apply max pooling
pooling_layer_output = torch.max(sumprod)

#%% pass preceding output through linear activation
intermediate_output_value = pooling_layer_output * lin_w + lin_b

#%% pass the output through the sigmoid operation
from torch.nn import functional as F
print(F.sigmoid(intermediate_output_value))




#%%

X_train[0, 0, 0:3, 0:3]








# %%
