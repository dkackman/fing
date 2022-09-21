import torch
from .service.settings import resolve_path

__version__ = "0.3.0"


def info():
    return {
        "software": {
            "name": "fing",
            "version": __version__,
            "torch_version": torch.__version__,
        }
    }
