#%%
import os
from torch_snippets import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob


#%%
IMAGE_ROOT = 'images/images'
DF_RAW = df = pd.read_csv('df.csv')

#%% define indices corresponding to labels and targets
label2target = {l:t+1 for t,l in enumerate(DF_RAW['LabelName'].unique())}
label2target['background'] = 0
target2label = {t:l for l,t in label2target.items()}
background_class = label2target['background']
num_classes = len(label2target)

#%% define the func to preprocess image
def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    return img.to(device).float()

#%% define dataset class - OpenDataset
class OpenDataset(Dataset):
    w, h = 224, 224
    def __init__(self, df, image_dir=IMAGE_ROOT):
        self.image_dir = image_dir
        self.files = glob.glob(self.image_dir + '/*')
        self.df = df
        self.image_infos = df.ImageID.unique()
        
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = find(image_id, self.files)
        img = Image.open(img_path).convert('RGB')
        data = df[df['ImageID'] == image_id]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute coordinates
        data[:, [0,2]] *= self.w
        data[:, [1,3]] *= self.h
        boxes = data.astype(np.uint32).tolist()
        # torch FRCNN EXPECTS ground truths as a dict of tensors
        target = {}
        target['boxes'] = torch.Tensor(boxes).float()
        target['labels'] = torch.Tensor([label2target[i] for i in labels]).long()
        img = preprocess_image(img)
        return img, target
        



