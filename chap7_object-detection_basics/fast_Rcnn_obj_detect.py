
#%%
from torch_snippets import *


#%% define dataset class that returns images, labels, ground truths
# region proposals and delta corresponding to each region proposal

class FRCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
    def __len__(self): return len(self.fpaths)
    
    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[...,::-1]
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        assert len(rois) == len(labels) == len(deltas), \
            f'{len(rois)}, {len(labels)}, {len(deltas)}'
        return image, rois, labels, deltas, gtbbs, fpath
    
    
    def collate_fn(self, batch):
        input, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(batch):
            image, image_rois, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            image = cv2.resize(image, (224,224))
            input.append(preprocess_image(image/255.)[None])
            rois.extend(image_rois)
            rixs.extend([ix]*len(image_rois))
            labels.extend([label2target[c] for c in image_labels])
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        rois = torch.Tensor(rois).float().to(device)
        rixs = torch.Tensor(rixs).float().to(device)
        labels = torch.Tensor(deltas).float().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, rois, rixs, labels, deltas
    
    
#%% create training and test datasets
n_train = 9 * len(FPATHS) // 10
train_ds = FRCNNDataset(FPATHS[:n_train], ROIS[:n_train],
                        CLSS[:n_train], DELTAS[:n_train],
                        GTBBS[:n_train]
                        )
test_ds = FRCNNDataset(FPATHS[n_train:], ROIS[n_train:],
                       CLSS[n_train:], DELTAS[n_train:],
                       GTBBS[n_train:]
                       )
        


#%% 
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import RoIPool

train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=2,collate_fn=test_ds.collate_fn, drop_last=True)

#%% 
class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        rawnet = torchvision.models.vgg16_bn(pretrained=True)
        for param in rawnet.features.parameters():
            param.requires_grad = True
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = RoIPool(7, spatial_scale=14/224)
        feature_dim = 512*7*7
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        self.bbox = nn.Sequential(nn.Linear(feature_dim, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 4),
                                  nn.Tanh()
                                  )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
        
    def forward(self, input, rois, ridx):
        res = input
        res = self.seq(res)
        rois = torch.cat([ridx.unsqueeze(-1), rois*224], dim=-1)
        res = self.roipool(res, rois)
        feat = res.view(len(res), -1)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
    
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs = torch.where(labels != background_class)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), \
                regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, \
                detection_loss.detach(), regression_loss
                
                
    
#%%  ## train batch ###
def train_batch(inputs, model, optimizer, criterion):
    input, rois, rixs, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input, rois, rixs)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()


def validate_batch(inputs, model, criterion):
    input, rois, rixs, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss, _deltas = model(input, rois, rixs)
        loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
        _clss = decode(_clss)
        accs  = clss == _clss
    return _clss, _deltas, loss.detach(), loc_loss, regr_loss, \
        accs.cpu().numpy()
        
        
#%%
frcnn = FRCNN().to(device)
criterion = frcnn.calc_loss
optimizer = optim.SGD(frcnn.parameters(), lr=1e-3)
n_epochs = 5
log = Report(n_epochs)
for epoch in range(n_epochs):
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, loc_loss, regr_loss, accs = train_batch(inputs, frcnn, optimizer, criterion)
        pos = (epoch + (ix + 1) / _n)
        log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss,
                   trn_regr_loss=regr_loss, trn_acc=accs.mean(),
                   end='\r'
                   )
        
    _n = len(test_loader)
        
    for ix, inputs in enumerate(test_loader):
        _clss, _deltas, loss, loc_loss, regr_loss, \
            accs = validate_batch(inputs, frcnn, criterion)
        pos = (epoch + (ix+1)/_n)
        log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss,
                   val_regr_loss = regr_loss,
                   val_acc=accs.mean(), end='\r'
                   )
log.plot_epochs('trn_loss, val_loss'.split(','))
        
        
#%%  define function to predict on test images 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.ops import nms
from PIL import Image


#%%
def test_predictions(filename):
    img = cv2.resize(np.array(Image.open(filename)), (224,224))
    candidates = extract_candidates(img)
    candidates = [(x,y,x+w,y+h) for x,y,w,h in candidates]
    input = preprocess_image(img/255.)[None]
    rois = [[x/224, y/224, X/224, Y/224] for x,y,X,Y in candidates]
    rixs = np.array([0] * len(rois))
    rois, rixs = [torch.Tensor(item).to(device) for item in [rois, rixs]]
    with torch.no_grad():
        frcnn.eval()
        probs, deltas = frcnn(input, rois, rixs)
        confs, clss = torch.max(probs, -1)
    candidates = np.array(candidates)
    confs, clss, probs, deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, probs, deltas]]
    bbs = candidates + deltas
    
    ixs = nms(torch.tensor(bbs.astype(np.float32)),
              torch.tensor(confs), 0.05)
    confs, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in 
                                                   [confs, clss, probs, deltas, candidates, bbs]
                                                   ]
    if len(ixs) == 1:
        confs, clss, probs, deltas, candidates, bbs = [tensor[None] for tensor in
                                                       [confs, clss, probs, deltas, candidates, bbs]
                                                       ]
    bbs = bbs.astype(np.uint16)
    
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    show(img, ax=ax[0])
    ax[0].grid(False)
    ax[0].set_title(filename.split('/')[-1])
    if len(confs) == 0:
        ax[1].imshow(img)
        ax[1].set_title('No objects')
        plt.show()
        return
    else:
        show(img, bbs=bbs.tolist(), texts=[target2label[c] for 
                                           c in clss.tolist()], ax=ax[1]
             )
        plt.show()
        
        
        
#%%
test_predictions(test_ds[29][-1])







