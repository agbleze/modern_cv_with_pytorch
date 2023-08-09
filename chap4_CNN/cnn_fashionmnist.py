
#%%
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam
from torchsummary import summary
import matplotlib.ticker as mticker


#%%   #### 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
data_folder = 'data/fmnist'

#%% get data

fmnist = datasets.FashionMNIST(root=data_folder, train=True, download=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

fmnist_val = datasets.FashionMNIST(root=data_folder, train=False, download=True)
val_images = fmnist_val.data
val_targets = fmnist_val.targets

#%% define dataset class
class FashionDataset(Dataset):
    def __init__(self, x,y):
        x = x.float() / 255
        x = x.view(-1, 1, 28, 28)
        self.x, self.y = x, y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, ix):
        x = self.x[ix]
        y = self.y[ix]
        return x.to(device), y.to(device)
    
#%% get data
def get_data(batch_size=32):
    tr_data = FashionDataset(tr_images, tr_targets)
    tr_dl = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
    
    val_data = FashionDataset(val_images, val_targets)
    val_dl = DataLoader(val_data, batch_size=len(val_images), shuffle=False)
    
    return tr_dl, val_dl
    
        
#%% define CNN model

def get_model():
    model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=3200, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=10)
            ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


#%% summarize model architecture
model, loss_fn, optimizer = get_model()
summary(model, torch.zeros(1,1,28,28))

#%%
@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()


#%% define accuracy function
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


#%% define training function
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()



#%%  define fun to trigger training
def trigger_train_process(train_dataload, val_dataload,
                          model, loss_fn, optimizer,
                          num_epochs=10
                          ):
    train_loss, train_accuracy = [], []
    valid_loss, valid_accuracy = [], []
    
    for epoch in range(num_epochs):
        print(epoch)
        train_epoch_losses, train_epoch_accuracies = [], []
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
            train_epoch_losses.append(batch_loss)
            
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
            
        for ix, batch in enumerate(iter(val_dataload)):
            x, y = batch
            valid_losses = val_loss(x, y, model, loss_fn)
            val_is_correct = accuracy(x, y, model)
            
        train_loss.append(np.array(train_epoch_losses).mean())
        train_accuracy.append(np.array(train_epoch_accuracies).mean())
        valid_loss.append(valid_losses)
        valid_accuracy.append(np.array(val_is_correct).mean())
        
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
    
#%%

tr_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

#%%
cnn_train_res = trigger_train_process(train_dataload=tr_dl,val_dataload=val_dl,
                                      model=model, loss_fn=loss_fn, 
                                      optimizer=optimizer
                                      )


#%%
cnn_train_loss = cnn_train_res['train_loss']  
cnn_train_acc = cnn_train_res['train_accuracy']  
cnn_valid_loss = cnn_train_res['valid_loss']   
cnn_valid_acc = cnn_train_res['valid_accuracy']


#%%
plot_loss(train_loss=cnn_train_loss,
          valid_loss=cnn_valid_loss,
          num_epochs=10, 
          title=f'Training and validation loss with CNN'
          )

#%%
plot_accuracy(train_accuracy=cnn_train_acc,
          valid_accuracy=cnn_valid_acc,
          num_epochs=10, 
          title=f'Training and validation Accuracy with CNN'
          )


#%% translate image b/t -5 pixels to +5 and predict class
preds = []
ix = 24300
for px in range(-5,6):
    img = tr_images[ix]/255
    img = img.view(28,28)
    img2 = np.roll(img, px, axis=1)
    plt.imshow(img2)
    plt.show()
    img3 = torch.Tensor(img2).view(-1, 1, 28, 28).to(device)    
    np_output = model(img3).cpu().detach().numpy()
    preds.append(np.exp(np_output)/np.sum(np.exp(np_output)))

#%% plot probability of the classes across various translations
import seaborn as sns
fig, ax = plt.subplots(1,1, figsize=(12,10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds).reshape(11,10), annot=True,
            ax=ax, fmt='.2f', xticklabels=fmnist.classes,
            yticklabels=[str(i)+str('pixels') for i in range(-5, 6)],
            cmap='gray'
             )




#%%  ###### image augmentation  #######

# plot 1st img
plt.imshow(tr_images[0])

#%% scaling #####
import imgaug.augmenters as iaa

#%%

aug = iaa.Affine(scale=2)

#%%

plt.imshow(aug.augment_image(np.array(tr_images[0])))
plt.title('Scaled image')

#%% image translation #####
aug = iaa.Affine(translate_px=10)
img_to_aug = np.array(tr_images[0])
plt.imshow(img_to_aug)

#%%
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Translated image by 10 pixels')

#%% diff translation for diff axis
aug = iaa.Affine(translate_px={'x': 10, 'y': 2})
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Translation of 10 pixels and 2 pixels over rows')


#%% set fit_output = True to prevent lossing data after transformation
plt.figure(figsize=(20, 20))
plt.subplot(161)
plt.imshow(img_to_aug)
plt.title('Original image')

plt.subplot(162)
aug = iaa.Affine(scale=2, fit_output=True)
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Scaled image')

plt.subplot(163)
aug = iaa.Affine(translate_px={'x':10, 'y': 2}, fit_output=True)
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Translation of 10 pixels acros ncols and 2 over rows')

plt.subplot(164)
aug = iaa.Affine(rotate=30, fit_output=True)
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Rotating to 30 deg')

plt.subplot(165)
aug = iaa.Affine(shear=30, fit_output=True)
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Shear image by 30\N{DEGREE SIGN} (degrees)')

#%% cval param to specify new pixel values when fit_output = True
aug = iaa.Affine(rotate=30, fit_output=True, cval=255)
plt.imshow(aug.augment_image(img_to_aug))
plt.title('Rotation of image by 30\N{DEGREE SIGN}')

#%% specify a range for rotation

aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0,
                 mode='constant'
                 )
plt.imshow(aug.augment_image(img_to_aug))



#%%  ##### changing the brightness #####
# Multiply and Linearcontrast are augmentation techniques
# for solving different lighting condition problems

aug = iaa.Multiply(0.5)
plt.imshow(aug.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255
           )
plt.title('Pixels multiplied by 0.5')

#%% Linearcontrast
aug = iaa.LinearContrast(0.5)
plt.imshow(aug.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255)
plt.title('Pixel contrast by 0.5')

#%% Blurring image
aug = iaa.GaussianBlur(sigma=1)
plt.imshow(aug.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255)
plt.title('Gaussian blurring of image')


#%%  ### Adding noise  -- dropout, salt and pepper ###

plt.figure(figsize=(10,10))
plt.subplot(121)
aug = iaa.Dropout(p=0.2)
plt.imshow(aug.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255)
plt.title('Random 20% pixel dropout')

plt.subplot(122)
aug = iaa.SaltAndPepper(0.2)
plt.imshow(aug.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255)
plt.title('Random 20% salt and pepper noise')

#%%  ##### performing a sequence of augmentations
seq = iaa.Sequential([
        iaa.Dropout(p=0.2),
        iaa.Affine(rotate=(-30,30))
        ], random_order=True
                     )

plt.imshow(seq.augment_image(img_to_aug), cmap='gray',
           vmin=0, vmax=255)
plt.title('Image augmented using random order of 2 augmentations')


#%% data aug on a batch of img
##### Scenario 1: Augmenting  32 images, one at a time ###
# 1. specify aug to perform
from imgaug import augmenters as iaa

aug = iaa.Sequential([
    iaa.Affine(translate_px={'x': (-10,10)},
               mode='constant'
               )
        ])

#%% %%time
import time

#%%time
for i in range(32):
    aug.augment_image(np.array(tr_images[i]))


#%% scenario 2: Augmenting 32 images as a batch in 1 go
aug.augment_images(np.array(tr_images[:32]))


#%% using collate_fn to enable batch augmentation of dataset

class FMNISTDataset(Dataset):
    def __init__(self, x, y, aug=None):
        self.x, self.y = x, y
        self.aug = aug
        
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y
    def __len__(self):
        return len(self.x)
    
    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))
        if self.aug:
            ims = self.aug.augment_images(ims)
        ims = torch.tensor(ims)[:,None,:,:].to(device)/255
        classes = torch.tensor(classes).to(device)
        return ims, classes


train = FMNISTDataset(x=tr_images, y=tr_targets, 
                      aug=aug
                      )

trn_dl = DataLoader(train, batch_size=64, collate_fn=train.collate_fn,
                    shuffle=True
                    )



#%%
"""

cmap; supported values are ['Accent', 'Accent_r', 
'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 
'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 
'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 
'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
"""



#%%
tr_images[0].shape


#%%

np.array(tr_images[0]).shape
#%%
from scipy.spatial.qhull import QhullError




# %%
