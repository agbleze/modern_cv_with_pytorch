
#%%
!curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/Users/lin/Documents/python_venvs/modern_cv_with_pytorch/chap17_moving_model_to_prod/shirt.png;type=image/png"
# %%
import cv2
from fmnist import FMNIST

#%%
cv2.imread("shirt.png", 0)
# %%

fmnist_init = FMNIST()

#%%

fmnist_init.predict(path='shirt2.png')




# %%
