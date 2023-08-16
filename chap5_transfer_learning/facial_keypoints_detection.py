
#%%
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torchsummary import summary
import numpy as np, pandas as pd, os, glob, cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_dir = 'P1_Facial_Keypoints/data/training/'
all_img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
data=pd.read_csv('P1_Facial_Keypoints/data/training_frames_keypoints.csv')

#%% 
data

#%% define class for providing input and output datapoint for data loader
class FacesData(Dataset):
    def __init__(self, df):
        super(FacesData, self).__init__()
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                              )
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ix):
        img_path = 'P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path) / 255
        
        kp = deepcopy(self.df.iloc[ix,1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[0]).tolist() 
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2
    
    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img.to(device)
    
    def load_img(self, ix):
        img_path='P1_Facial_Keypoints/data/training/' + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) / 255
        img = cv2.resize(img, (224, 224))
        return img
    
    
#%% ## training and test data split
train, test = train_test_split(data, test_size=0.2, 
                               random_state=101
                               )
train_dataset = FacesData(train.reset_index(drop=True))
test_dataset = FacesData(test.reset_index(drop=True))

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
            

#%% ### VGG16 MODEL ###
vgg16_model = models.vgg16(pretrained=True)
def get_model(pretrained_model=vgg16_model):
    model = pretrained_model
    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.Sequential(
                        nn.Conv2d(512, 512, 3),
                        nn.MaxPool2d(2),
                        nn.Flatten()
                )
    
    model.classifier = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 136),
                    nn.Sigmoid()
                )
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), criterion, optimizer


#%% #### 
model, criterion, optimizer = get_model()

#%% 
def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

#%% loss on test data and predicted key points
def validate_batch(img, kps, model, criterion):
    model.eval()
    with torch.no_grad():
        _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss

#%% 
def trigger_train_process(train_dataload, val_dataload,
                          model, criterion, optimizer,
                          n_epochs=50):
    train_loss, test_loss = [], []
    
    for epoch in range(n_epochs):
        print(f'epoch {epoch+1} : 50')
        epoch_train_loss, epoch_test_loss = 0, 0
        for ix, (img, kps) in enumerate(train_dataload):
            loss = train_batch(img, kps, model, optimizer, 
                               criterion
                               )
            epoch_train_loss += loss.item()
        epoch_train_loss /= (ix + 1)
        
        for ix, (img, kps) in enumerate(val_dataload):
            ps, loss = validate_batch(img, kps, model, criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= (ix + 1)
        
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)
    return {'train_loss': train_loss,
            'test_loss': test_loss
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
vgg_fkp_train_res = trigger_train_process(train_dataload=train_loader,
                                          val_dataload=test_loader,
                                          model=model,
                                          criterion=criterion,
                                          optimizer=optimizer
                                          )

#%%
train_loss = vgg_fkp_train_res['train_loss']
test_loss = vgg_fkp_train_res['test_loss']

plot_loss(train_loss=train_loss, valid_loss=test_loss,
          num_epochs=50, title='Facial keypoints train and test loss for VGG16'
          )
#%%  #### test model on random test image index
ix = 0
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.title('Original image')
im = test_dataset.load_img(ix)
plt.imshow(im)
plt.grid(False)
plt.subplot(222)
plt.title('Image with facial keypoints')
x, _ = test_dataset[ix]
kp = model(x[None]).flatten().detach().cpu()
plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
plt.grid(False)
plt.show()


#%% 




