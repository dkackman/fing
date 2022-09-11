from flask import Flask
from flask_restful import Api
import logging
import torch
import worker
from Resources import txt2imgResource, txt2imgMetadataResource, InfoResource
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

def create_app(model_name, auth_token):
    if not torch.cuda.is_available():
        logging.critical("CUDA not available")
        raise Exception("unavailable")  # don't try to run this on cpu
    
    logging.debug(f"Torch version {torch.__version__}")

    # load the model into the gpu - stays there for the life of the process
    worker.load_model(model_name, auth_token)

    app = Flask("txt2img service")
    api = Api(app)

    api.add_resource(InfoResource, '/info')
    api.add_resource(txt2imgResource, '/txt2img')
    api.add_resource(txt2imgMetadataResource, '/txt2img_metadata')

    return app

def setup_logging(config):
    log_path = config.resolve_path(config.config_file["generation"]["log_filename"])
    log_date_format = "%Y-%m-%dT%H:%M:%S"
    
    logger = logging.getLogger()

    handler = ConcurrentRotatingFileHandler(log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt=log_date_format,
        )
    )
    logger.setLevel(logging.INFO)
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
