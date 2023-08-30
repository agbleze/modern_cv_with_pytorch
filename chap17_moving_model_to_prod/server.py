import os, io
from fmnist import FMNIST
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile

model = FMNIST()

app = FastAPI()

@app.post("/predict")
def predict(request: Request, file: UploadFile=File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert('L')
    output = model.predict(image)
    return output

#%%
#import numpy as np


#np.__version__
# %%
