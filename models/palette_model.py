from pydantic import BaseModel

class PaletteResponse(BaseModel):
    string: list[str]