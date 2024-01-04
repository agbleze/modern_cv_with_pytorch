
from torch_snippets import *


#%%
!wget -O train-annotations-object-segmentation.csv -q https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv
!wget -O classes.csv -q \
https://raw.githubusercontent.com/openimages/dataset/master/dict.csv
# %%
required_classes = 'person,dog,bird,car,elephant,football,\
jug,laptop,Mushroom,Pizza,Rocket,Shirt,Traffic sign,\
Watermelon,Zebra'

required_classes = [c.lower() for c in required_classes.lower().split(",")]

classes = pd.read_csv('classes.csv', header=None)
classes.columns = ['class', "class_name"]
classes = classes[classes['class_name'].map(lambda x: x in required_classes)]

