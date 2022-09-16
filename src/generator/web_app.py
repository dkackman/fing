from flask import Flask
from flask_restful import Api
import logging
import torch
from gpu import Gpu
from InfoResource import InfoResource
from txt2imgResource import txt2imgResource, txt2imgMetadataResource
from img2imgResource import img2imgResource, img2imgMetadataResource
from imginpaintResource import imginpaintResource, imginpaintMetadataResource
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pipelines import preload_pipelines


def create_app(model_name, auth_token):
    if not torch.cuda.is_available():
        logging.critical("CUDA not available")
        raise Exception("CUDA unavailable")  # don't try to run this on cpu
    
    logging.debug(f"Torch version {torch.__version__}")

    default_device = Gpu()
    # load the model into ram - stays there for the life of the process
    preload_pipelines(model_name, auth_token)

    app = Flask("stable-diffusion service")
    api = Api(app)

    api.add_resource(InfoResource, '/info', resource_class_kwargs={ 'model': default_device })
    api.add_resource(txt2imgResource, '/txt2img', resource_class_kwargs={ 'model': default_device })

    api.add_resource(txt2imgMetadataResource, '/txt2img_metadata', resource_class_kwargs={ 'model': default_device })

    api.add_resource(img2imgResource, '/img2img', resource_class_kwargs={ 'model': default_device })
    api.add_resource(img2imgMetadataResource, '/img2img_metadata', resource_class_kwargs={ 'model': default_device })

    api.add_resource(imginpaintResource, '/imginpaint', resource_class_kwargs={ 'model': default_device })
    api.add_resource(imginpaintMetadataResource, '/imginpaint_metadata', resource_class_kwargs={ 'model': default_device })

    return app

def setup_logging(config):
    log_path = config.resolve_path(config.config_file["generation"]["log_filename"])
    log_date_format = "%Y-%m-%dT%H:%M:%S"
    
    logger = logging.getLogger()

    handler = ConcurrentRotatingFileHandler(log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=log_date_format)
    )
    logger.setLevel(logging.INFO) # unless otherwise specified leglevel will be info
    logging_config = config.config_file["generation"]
    if "log_level" in logging_config:
        if logging_config["log_level"] == "CRITICAL":
            logger.setLevel(logging.CRITICAL)
        elif logging_config["log_level"] == "ERROR":
            logger.setLevel(logging.ERROR)
        elif logging_config["log_level"] == "WARNING":
            logger.setLevel(logging.WARNING)
        elif logging_config["log_level"] == "DEBUG":
            logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
