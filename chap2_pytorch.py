
#%%
import torch

#%%
x = torch.tensor([[2., -1.],[1., 1.]], requires_grad=True)
print(x)

out = x.pow(2).sum()

#%% cal gradient
out.backward()
x.grad

# %%
