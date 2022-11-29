from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from deep_draw.interface.app import image_bytes, bitmap_normalized

app = FastAPI()

@app.get("/predict")
async def main(data='data'):
    prediction = predict(data)
    return {'prediction' : prediction}
