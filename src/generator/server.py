import torch
from .settings import load_settings
from .service.web_app import create_app
from . import setup_logging, __version__
import uvicorn


if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")


settings = load_settings()
setup_logging(settings.log_filename, settings.log_level)

app = create_app(
    settings.model_name, settings.huggingface_token, settings.model_cache_dir
)


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
