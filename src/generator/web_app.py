from flask import Flask
from flask_restful import Api
import logging
import torch
from gpu import Gpu
from InfoResource import InfoResource
from txt2imgResource import txt2imgResource, txt2imgMetadataResource
from img2imgResource import img2imgResource, img2imgMetadataResource
from imginpaintResource import imginpaintResource, imginpaintMetadataResource
from pipelines import Pipelines


def create_app(model_name, auth_token):
    if not torch.cuda.is_available():
        logging.critical("CUDA not available")
        raise Exception("CUDA unavailable")  # don't try to run this on cpu
    
    logging.debug(f"Torch version {torch.__version__}")

    pipelines = Pipelines(model_name)
    pipelines.preload_pipelines(auth_token)
    default_device = Gpu(pipelines)

    app = Flask("stable-diffusion service")
    api = Api(app)

    api.add_resource(InfoResource, '/info')

    api.add_resource(txt2imgResource, '/txt2img', resource_class_kwargs={ 'device': default_device })
    api.add_resource(txt2imgMetadataResource, '/txt2img_metadata', resource_class_kwargs={ 'device': default_device })

    api.add_resource(img2imgResource, '/img2img', resource_class_kwargs={ 'device': default_device })
    api.add_resource(img2imgMetadataResource, '/img2img_metadata', resource_class_kwargs={ 'device': default_device })

    api.add_resource(imginpaintResource, '/imginpaint', resource_class_kwargs={ 'device': default_device })
    api.add_resource(imginpaintMetadataResource, '/imginpaint_metadata', resource_class_kwargs={ 'device': default_device })

    return app
