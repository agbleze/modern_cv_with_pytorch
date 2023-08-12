
#%%
import torch
import torch.nn as nn
import numpy as np, cv2, pandas as pd, glob, time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms, datasets
from matplotlib import ticker as mticker


#%% fetch dataset




#%%  ##   #####
trn_df = pd.read_csv('fairface-label-train.csv')
val_df = pd.read_csv('fairface-label-val.csv')
trn_df.head()

device = 'cuda' if torch.is_available() else 'cpu'

#%% prepare dataset 
IMAGE_SIZE = 224
class GenderAgeClass(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                              )
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = f.file
        gen = f.gender == 'Female'
        age = f.age
        im = cv2.imread(file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen
    
    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im/255).permute(2,0,1)
        im = self.normalize(im)
        return im[None]
    
    def collate_fn(self, batch):
        ims, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            ims.append(im)
            ages.append(float(int(age))/80)
            genders.append(float(gender))
        ages, genders = [torch.tensor(x).to(device).float()
                         for x in [ages, genders]
                         ]
        ims = torch.cat(ims).to(device)
        return ims, ages, genders


#%% 
trn = GenderAgeClass(trn_df)
val = GenderAgeClass(val_df)
train_loader = DataLoader(trn, batch_size=32, shuffle=True,
                          drop_last=True, 
                          collate_fn=trn.collate_fn
                          )
test_loader = DataLoader(val, batch_size=32, collate_fn=val.collate_fn,)
a,b,c = next(iter(train_loader))
print(a.shape, b.shape, c.shape)

#%%
def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
                            nn.Conv2d(512, 512, kernel_size=3),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Flatten()
                        )
    
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                                        nn.Linear(2048, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.4),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Dropout(0.4),
                                        nn.Linear(128, 64),
                                        nn.ReLU()
                                    )
            self.age_classifier = nn.Sequential(
                                        nn.Linear(64, 1),
                                        nn.Sigmoid()
                                    )
            self.gender_classifier = nn.Sequential(
                                        nn.Linear(64, 1),
                                        nn.Sigmoid()
                                    )
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age
    
    model.classifier = ageGenderClassifier()
    
    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    loss_functions = gender_criterion, age_criterion
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), loss_functions, optimizer


#%% 
model, criterion, optimizer = get_model()

#%% ### 
def train_batch(data, model, optimizer, criterion):
    model.train()
    ims, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(ims)
    gender_criterion, age_criterion = criterion
    gender_loss = gender_criterion(pred_gender.squeeze(), gender)
    age_loss = age_criterion(pred_age.squeeze(), age)
    total_loss = gender_loss + age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss

#%%
def validate_batch(data, model, criteria):
    ims, age, gender = data
    model.eval()
    with torch.no_grad():
        pred_gender, pred_age = model(ims)
        gender_criterion, age_criterion = criteria
        gender_loss = gender_criterion(pred_gender.squeeze(), gender)
        age_loss = age_criterion(pred_age.squeeze(), age)
        total_loss = gender_loss + age_loss
        pred_gender = (pred_gender > 0.5).squeeze()
        gender_acc = (pred_gender == gender).float().sum()
        
        age_mae = torch.abs(age - pred_age).float().sum()
        return total_loss, gender_acc, age_mae
    
    
#%% train the model over 5 epochs
import time

model, criteria, optimizer = get_model()
val_gender_accuracies = []
val_age_maes = []
train_losses = []
val_losses = []
n_epochs = 5
best_test_loss = 1000
start = time.time()

for epoch in range(n_epochs):
    epoch_train_loss, epoch_test_loss = 0, 0
    val_age_mae, val_gender_acc, ctr = 0,0,0
    _n=len(train_loader)
    for ix, data in enumerate(train_loader):
        loss = train_batch(data, model, optimizer, criteria)
        epoch_train_loss += loss.item()
        
    for ix, data in enumerate(test_loader):
        loss, gender_acc, age_mae = validate_batch(data, model, criteria)
        epoch_test_loss += loss.item()
        val_age_mae += age_mae
        val_gender_acc += gender_acc
        ctr += len(data[0])
        
    # cal overall accuracy of age pred and gender classifications
    val_age_mae /= ctr
    val_gender_acc /= ctr
    epoch_train_loss /= len(train_loader)
    epoch_test_loss /= len(test_loader)
        
    #log metrics of each epoch
    elapsed = time.time() -start
    best_test_loss = min(best_test_loss, epoch_test_loss)
    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(epoch+1, n_epochs, time.time()-start,
                                                       (n_epochs-epoch)*(elapsed/epoch+1)
                                                       )
          )
    info = f'''Epoch: {epoch+1:03d}\tTrain Loss: {epoch_train_loss:.3f}
                \tTest: {epoch_test_loss:.3f}
                \tBest test Loss: {best_test_loss:.4f}
            '''
    info += f'\nGender Accuracy: {val_gender_acc*100:.2f}%\tAge MAE: \
                {val_age_mae:.2f}\n'
    print(info)
    
    val_gender_accuracies.append(val_gender_acc)
    val_age_maes.append(val_age_mae)
        
        

    
    
    ## system application and product in data processing




# %%
