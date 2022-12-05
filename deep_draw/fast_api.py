from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_draw.interface.main import pred
import tensorflow as tf
import json
from pydantic import BaseModel
import numpy as np
from deep_draw.dl_logic.utils import image_from_dict, image_from_dict_RNN

class Item(BaseModel):
    image: str
    size: int
    height : int
    width : int
    channel : int

class Item_RNN(BaseModel):
    image: str
    size: int
    height : int
    width : int

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict/")
async def image(item: Item):
    np_array_image = image_from_dict(dict(item))
    prediction, first_5_stats = pred((np_array_image/255.), select_model='cnn')
    return {'test' : prediction, "class" : first_5_stats}


@app.post("/predictRNN/")
async def image(item: Item_RNN):
    np_array_image = image_from_dict_RNN(dict(item))
    prediction, first_5_stats = pred(np_array_image, select_model='rnn')
    return {'test' : prediction, "class" : first_5_stats}


@app.get("/")
def index():
    return {'ok' : "jusqu'ici tout va bien"}
