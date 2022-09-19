import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import torch
from .config import Config

__version__ = "0.3.0"


def setup_logging(config):
    log_path = Config.resolve_path(config.config_file["generation"]["log_filename"])
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


def info():
    return {
            'software': {
                'name': "fing",
                'version': __version__,
                'torch_version': torch.__version__,
            }
        }
