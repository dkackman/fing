from flask import Flask
from flask_restful import Api
import logging
import torch
import worker
from Resources import txt2imgResource, txt2imgMetadataResource, InfoResource

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
    api.add_resource(txt2imgMetadataResource, '/txt2img_metadaa')

    return app
