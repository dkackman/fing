import torch
from .service.web_app import create_app
from . import __version__
import uvicorn


if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")

app, settings = create_app()


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
