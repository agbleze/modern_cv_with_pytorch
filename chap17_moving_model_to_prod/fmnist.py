#%%
from torch_snippets import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FMNIST(nn.Module):
    classes = ['T-shit/top', 'Trouser', 'Pullover', 'Dress',\
             'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
             ]
    def __init__(self):
    self.model = nn.Sequential()
    