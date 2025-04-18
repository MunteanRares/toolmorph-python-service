import tempfile

import uvicorn
from fastapi import FastAPI, UploadFile, File
from uvicorn import logging

from services.palette_extractor import PaletteExtractor
from models import palette_model
import os

app = FastAPI()

@app.post("/extract-palette")
async def extract_palette(file: UploadFile = File(...)):
    print("smth")
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.content_type) as tmp:
        tmp_name = tmp.name
        file_content = await file.read()
        tmp.write(file_content)

    extractor = PaletteExtractor(tmp_name)
    palette = extractor.extract_palette()
    os.remove(tmp_name)

    return {"palette": palette}