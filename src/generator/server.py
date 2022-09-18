import sys
import logging
from generator.config import Config
from generator.service.web_app import create_app
from . import setup_logging


config = Config().load()
setup_logging(config)

if __name__ == "__main__":
    app = create_app(config.config_file["model"]["model_name"],
        config.config_file["model"]["huggingface_token"],
        config.config_file["generation"]["x_api_key_enabled"],
        config.config_file["generation"]["x_api_key_list"],
    )
    host = config.config_file["generation"]["host"]
    port = config.config_file["generation"]["port"]

    logging.info(f'Starting server at {host}:{port}')        
    app.run(host, port)    
    logging.info("Server exiting")

else:
    logging.info("Starting with WSGI server")
    gunicorn_app = create_app(config.config_file["model"]["model_name"],
            config.config_file["model"]["huggingface_token"],
            config.config_file["generation"]["x_api_key_enabled"],
            config.config_file["generation"]["x_api_key_list"],
        )
