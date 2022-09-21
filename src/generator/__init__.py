import torch
from pydantic import BaseModel

__version__ = "0.3.0"


class Software(BaseModel):
    name: str
    version: str
    torch_version: str


class InfoModel(BaseModel):
    software: Software


def info() -> InfoModel:
    return InfoModel(
        software=Software(
            name="fing", version=__version__, torch_version=torch.__version__
        )
    )
