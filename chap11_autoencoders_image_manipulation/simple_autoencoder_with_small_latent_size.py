
#%%
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

img_transform = transforms.Compose(
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.to(device))
)


# create train and val datasets
trn_ds = MNIST(root="/content/", transform=img_transform,
                train=True, download=False
                )

val_ds = MNIST("/content/", transform=img_transform,train=False, download=True)

#%%
batch_size = 256
trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latend_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=latent_dim)
        )

