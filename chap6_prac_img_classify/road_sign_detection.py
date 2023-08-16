#%%
import os
from torch_snippets import *
from torchvision import transforms as T

#%%
classIds = pd.read_csv('signnames.csv')
classIds.set_index('ClassId', inplace=True)
classIds = classIds.to_dict()['SignName']
classIds = {f'{k:05d}':v for k,v in classIds.items()}
id2int = {v:ix for ix, (k,v) in enumerate(classIds.items())}

#%%
trn_tfms = T.Compose([
    T.ToPILImage(), T.Resize(32), T.CenterCrop(32),
    # T.ColorJitter(brightness=(0.8, 1.2),
    #               contrast=(0.8, 1.2),
    #               saturation=(0.8, 1.2),
    #               hue=0.25 
    #               ),
    
    # T.RandomAffine(5, translate=(0.01, 0.1)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
])


val_tfms = T.Compose([
    T.ToPILImage(), T.Resize(32), T.CenterCrop(32),T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
])


#%%  
class GTSRB(Dataset):
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
        return img, classIds[clss]
    
    def choose(self):
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        imgs, classes = list(zip(*batch))
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        classes = [torch.tensor([id2int[clss] for clss in classes])]
        imgs, classes = [torch.cat(i).to(device) for i in [imgs, classes]]
        return imgs, classes

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
all_files = Glob('GTSRB/Final_Training/Images/*/*.ppm')
np.random.seed(10)
np.random.shuffle(all_files)

from sklearn.model_selection import train_test_split

trn_files, val_files = train_test_split(all_files, random_state=1)

trn_ds = GTSRB(trn_files)
trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True, collate_fn=trn_ds.collate_fn)

val_ds = GTSRB(val_files)
val_dl = DataLoader(val_ds, batch_size=32, 
                    shuffle=True, 
                    collate_fn=val_ds.collate_fn
                    )

#%%
import torchvision.models as models

def convBlock(ni, no):
    return nn.Sequential(nn.Dropout(0.2),
                         nn.Conv2d(ni, no, kernel_size=3,padding=1),
                         nn.ReLU(inplace=True),
                         ## nn.BatchNorm2d(no),
                         nn.MaxPool2d(2),
                         )
    
class SignClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(convBlock(3, 64),
                                   convBlock(64, 64),
                                   convBlock(64, 128),
                                   convBlock(128, 64),
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
        ce_loss = self.loss_fn(preds, targets)
        acc = (torch.max(preds, 1)[1]==targets).float().mean()
        return ce_loss, acc
    
    
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, labels = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


def validate_batch(model, data, criterion):
    model.eval()
    ims, labels = data
    _preds = model(ims)
    loss, acc = criterion(_preds, labels)
    return loss.item(), acc.item()


model = SignClassifier().to(device)
criterion = model.compute_metrics
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 50
log = Report(n_epochs)

for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, end='\r')
    
    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        log.record(ex+(bx+1)/N, val_loss=loss, val_acc=acc, end='\r')
        log.report_avgs(ex+1)
        
    if ex == 10:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

log.plot_epochs()
dumpdill(log, 'no-aug-no-bn.log')    

        





