from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from app import image_bytes, bitmap_normalized

app = FastAPI()

@app.get("/predict")
async def main():

    return prediction
