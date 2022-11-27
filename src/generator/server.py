import logging
import torch
import uvicorn
from . import __version__
from .settings import (
    load_settings,
    resolve_path,
    settings_exist,
)
from .log_setup import setup_logging
from .service.web_app import create_app
from .diffusion.device import Device
from .diffusion.device_pool import add_device_to_pool


if not settings_exist():
    raise Exception("no settings run 'python -m generator.init_app'")

settings = load_settings()


app = create_app(__version__, settings.x_api_key_enabled, settings.x_api_key_list)


@app.on_event("startup")
async def startup_event():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Torch version {torch.__version__}")

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i, settings.huggingface_token))


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
