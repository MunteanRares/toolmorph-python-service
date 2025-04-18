from pydantic import BaseModel
from typing import List

class PaletteResponse(BaseModel):
    palettes: List[str]