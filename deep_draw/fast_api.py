from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_draw.interface.main import pred
import tensorflow as tf
import json
from pydantic import BaseModel
import numpy as np
from deep_draw.dl_logic.utils import image_from_dict

class Item(BaseModel):
    image: str
    size: int
    height : int
    width : int
    channel : int

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# @app.post("/predict/")
# async def main(img: UploadFile = File(...)):
#     img.filename = "content"
#     img_step = await img.read()  # <-- Important!

#     img_step=tf.io.decode_image(img_step,
#                            channels=3,
#                            dtype=tf.dtypes.float32,
#                            name=None,
#                            expand_animations=True)

#     return {'prediction' : "jusqu'ici tout va bien"}

@app.post("/predict/")
async def image(item: Item):
    np_array_image = image_from_dict(dict(item))
    prediction = pred(np_array_image)
    return {'test' : prediction}


@app.get("/")
def index():
    return {'ok' : "jusqu'ici tout va bien"}
