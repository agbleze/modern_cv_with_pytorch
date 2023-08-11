
#%%
import face_alignment, cv2
from torch_snippets import read, show

#%%
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  flip_input=False,
                                  device='cpu'
                                  )

input = cv2.imread('Hema.JPG', 1)
preds = fa.get_landmarks(input)[0]
print(preds.shape)

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,5))
show(read('Hema.JPG'))
# plt.imshow(cv2.cvtColor(cv2.imread('Hema.JPG')), 
#            cv2.COLOR_BGR2RGB
#            )
ax.scatter(preds[:,0], preds[:,1],
           marker='+', c='r'
           )
plt.show()




# %%
