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

@app.post("/predict/")
async def image(item: Item):
    np_array_image = image_from_dict(dict(item))
    prediction, first_5_stats = pred((np_array_image/255.))
    return {'test' : prediction, "class" : first_5_stats}


@app.get("/")
def index():
    return {'ok' : "jusqu'ici tout va bien"}
