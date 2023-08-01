#%%
import cv2, matplotlib.pyplot as plt

img = cv2.imread('Hemanvi.jpeg')

#%%
print(img)

#%%
img = img[50:250, 40:240, :]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
print(img.shape)

#%% bottom-right 3x3 array of pixels
crop = img[-3:, -3:]
print(crop)
plt.imshow(crop)





# %%
