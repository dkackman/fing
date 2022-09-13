import sys
from config import Config
import logging
from web_app import create_app, setup_logging
import logging

# TODO 
# - auth https://www.bacancytechnology.com/blog/flask-jwt-authentication
# - manage multiple gpus

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
