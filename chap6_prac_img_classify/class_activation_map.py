
#%% class acctivation maps
import os
from torch_snippets import *
from torchvision import transforms as T

#%% indices of output classes
id2int = {'Parasitized': 0, 'Uninfected': 1}

#%% transformations done on images
trn_tfms = T.Compose([
                        T.ToPILImage(),
                        T.Resize(128),
                        T.CenterCrop(128),
                        T.ColorJitter(brightness=(0.95, 1.05),
                                    contrast=(0.95,1.05),
                                    saturation=(0.95,1.05),
                                    hue=0.05
                                    ),
                        T.RandomAffine(5, translate=(0.01,0.1)),
                        T.ToTensor(), 
                        T.Normalize(mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5]
                                    )
                ])

# transformation for val dataset
val_tfms = T.Compose([
                        T.ToPILImage(),
                        T.Resize(128),
                        T.CenterCrop(128),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                    ])


#%% define the dataset class
class MalariaImages(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        logger.info(len(self))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, ix):
        fpath = self.files[ix]
        clss = fname(parent(fpath))
        img = read(fpath, 1)
        return img, clss
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        _imgs, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img)[None] for img in _imgs]
        classes = [torch.tensor([id2int[clss]]) for clss in classes]
        imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
        return imgs, classes, _imgs


#%% fetch train and val dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
all_files = Glob('cell_images/*/*.png')
np.random.seed(10)
np.random.shuffle(all_files)

from sklearn.model_selection import train_test_split

trn_files, val_files = train_test_split(all_files, random_state=1)

trn_ds = MalariaImages(trn_files, transform=trn_tfms)
val_ds = MalariaImages(val_files, transform=val_tfms)

trn_dl = DataLoader(trn_ds, 32, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, 32, shuffle=False, collate_fn=val_ds.collate_fn)


#%% define the model
def convBlock(ni, no):
    return nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Conv2d(ni, no, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(no),
                    nn.MaxPool2d(2)
                )
    
#%% 
class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            convBlock(3, 64),
            convBlock(64, 64),
            convBlock(64, 128),
            convBlock(128, 256),
            convBlock(256, 512),
            convBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(id2int))
            
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def compute_metrics(self, preds, targets):
        loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1]==targets).float().mean()
        return loss, acc


#%%
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, labels, _ = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

#%%
@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, labels, _ = data
    _preds = model(ims)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()


#%%
model = MalariaClassifier().to(device)
criterion = model.compute_metrics
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 2
log = Report(n_epochs)

#%%
