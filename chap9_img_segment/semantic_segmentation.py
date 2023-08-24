
#%%
import os
if not os.path.exists('dataset1'):
    !wget -q hhtps://www.
    !unzip -q dataset1.zip
    !rm dataset1.zip
    !pip install -q torch_snippets pytorch_model_summary
    
#%%
from torch_snippets import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225]
                                                )
                           ]
                          )


#%%
class SegData(Dataset):
    def __init__(self, split):
        self.items = stem(f'dataset1/images_prepped_{split}')
        self.split = split
    def __len__(self):
        return len(self.items)
        





