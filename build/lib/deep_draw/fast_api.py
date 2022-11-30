from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from deep_draw.interface.main import pred

app = FastAPI()

@app.post("/predict")
async def main(img: UploadFile = File(...)):
    #prediction = pred(data)
    return {'prediction' : "jusqu'ici tout va bien"}


@app.get("/")
def index():
    return {'ok' : "jusqu'ici tout va bien"}
