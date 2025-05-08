import asyncio
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse

from services.palette_extractor import PaletteExtractor
from services.background_remover import BackgroundRemover
from models.palette_model import PaletteResponse
import os

app = FastAPI()

@app.post("/extract-palette", response_model=PaletteResponse)
async def extract_palette(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_name = tmp.name
        file_content = await file.read()
        tmp.write(file_content)

    try:
        palette = await asyncio.to_thread(PaletteExtractor(tmp_name).extract_palette)
    finally:
        os.remove(tmp_name)

    return {"palettes": palette}

@app.post("/background-remover")
async def remove_background(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp_name = tmp.name
        file_content = await file.read()
        tmp.write(file_content)

    try:
        remover = BackgroundRemover(tmp_name)
        byte_stream = await asyncio.to_thread(remover.remove_background)

    finally:
        os.remove(tmp_name)

    return StreamingResponse(byte_stream, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
    )