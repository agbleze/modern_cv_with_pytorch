#%%
from torch_snippets import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'