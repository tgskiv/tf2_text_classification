from typing import Union
from fastapi import FastAPI


import os
import sys


# relative path from folder with this script to absolute path
# model_dir = os.path.join(sys.path[0], '../tf2_text_classification')
# checkpoint_dir = os.path.join(sys.path[0], '../tf2_text_classification/checkpointsMy/')
# sys.path.insert(1, model_dir)
# import MovieReviewClassificationModel as mrcm

checkpoint_dir = os.path.join(sys.path[0], 'checkpointsMy/')
from lib import MovieReviewClassificationModel as mrcm


app = FastAPI()
model = None

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def load_model():
    global model
    model = mrcm.MovieReviewClassificationModel()

    model.load_weights(checkpoint_dir)
    model.compile_model()


@app.get("/review")
def review(q: str = None):
    print(q)

    if q is None:
        return {"error": "No review text provided"}
    else:
        return model.predict([q])


