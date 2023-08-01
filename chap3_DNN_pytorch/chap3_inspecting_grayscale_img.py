#%%
import cv2, matplotlib.pyplot as plt

img = cv2.imread('Hemanvi.jpeg')


#%%  ## crop image
img = img[50:250, 40:240]

# convert image to grayscale

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# show image
plt.imshow(img_gray, cmap='gray')

#%% convert image to 25 X 25 array
img_gray_small = cv2.resize(img_gray, (25, 25))
plt.imshow(img_gray_small, cmap='gray')


#%% # inspect pixel values
print(img_gray_small)





# %%
