
#%%
from torch_snippets import *
import selectivesearch
from skimage.segmentation import felzenszwalb

#%%
!wget https://www.dropbox.com/s/l98leemr7r5stnm/Hemanvi.jpeg
img = read('Hemanvi.jpeg', 1)

# %%
segments_fz = felzenszwalb(img, scale=200)

subplots([img, segments_fz], titles=['Original image','Image post felzenszwalb segmentation'],
         sz=10, nc=2
         )

#%% define func to fetch region proposals from image
def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates
    
#%% extract candidates and plot them on top on an image
candidates = extract_candidates(img)
show(img, bbs=candidates)

#%% func takes 2 bounding boxes as inputs and returns IoU
def get_iou(boxA, boxB, epsilon=1e-5):
    # cal coordinates of the intersection
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    
    # cal width and height
    width = (x2 - x1)
    height = (y2 - y1)
    
    # cal area of overlap
    if (width < 0 ) or (height < 0):
        return 0
    area_overlap = width * height
    
    # cal combined area of 2 bounding boxes
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    
    # cal iou
    iou = area_overlap / (area_combined + epsilon)
    return iou



# %%
