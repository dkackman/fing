import sys
import logging
from generator.config import Config
from generator.service.web_app import create_app
from . import setup_logging

# TODO #13 auth https://www.bacancytechnology.com/blog/flask-jwt-authentication

config = Config().load()
setup_logging(config)

if __name__ == "__main__":
    try:  
        app = create_app(config.config_file["model"]["model_name"], config.config_file["model"]["huggingface_token"])
        host = config.config_file["generation"]["host"]
        port = config.config_file["generation"]["port"]
        logging.info(f'Starting server at {host}:{port}')        
        app.run(host, port)
        
        logging.info("Server exiting")
    except:
        print("Fatal error", file=sys.stderr)
        sys.exit(1)
else:
    gunicorn_app = create_app(config.config_file["model"]["model_name"], config.config_file["model"]["huggingface_token"])
