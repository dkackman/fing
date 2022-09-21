import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
import torch
from .settings import resolve_path

__version__ = "0.3.0"


def setup_logging(log_path, log_level):
    log_path = resolve_path(log_path)
    log_date_format = "%Y-%m-%dT%H:%M:%S"

    logger = logging.getLogger()

    handler = ConcurrentRotatingFileHandler(
        log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=log_date_format
        )
    )

    if log_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    elif log_level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(
            logging.INFO
        )  # unless otherwise specified leglevel will be info

    logger.addHandler(handler)


def info():
    return {
        "software": {
            "name": "fing",
            "version": __version__,
            "torch_version": torch.__version__,
        }
    }
