#%%
from torch_snippets import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FMNIST(nn.Module):
    classes = ['T-shit/top', 'Trouser', 'Pullover', 'Dress',\
             'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
             ]
    def __init__(self, fpath = "fmnist.weights.pth"):
        super().__init__()
        self.model = nn.Sequential(
                            nn.Linear(28*28, 1000),
                            nn.ReLU(),
                            nn.Linear(1000, 10)
                        ).to(device)
        self.model.load_state_dict(torch.load(fpath))
        logger.info('Loaded FMNIST Model')
    
    @torch.no_grad()    
    def forward(self, x):
        x = x.view(1, -1).to(device)
        pred = self.model(x)
        pred = F.softmax(pred, -1)[0]
        conf, clss = pred.max(-1)
        clss = self.classes[clss.cpu().item()]
        return conf.item(), clss
    
    def predict(self, path):
        #print(path)
        image = cv2.imread(path, 0)
        x = np.array(image)
        x = cv2.resize(x, (28,28))
        x = torch.Tensor(255-x)/255.
        conf, clss = self.forward(x)
        return {'class': clss, 'confidence': f'{conf:.4f}'}
        
    
    
    
    