from flask import Flask
from flask_restful import Api
import logging
import torch
from generator.diffusion.device import Device
from generator.service.InfoResource import InfoResource
from generator.service.txt2imgResource import txt2imgResource, txt2imgMetadataResource
from generator.service.img2imgResource import img2imgResource, img2imgMetadataResource
from generator.service.imginpaintResource import imginpaintResource, imginpaintMetadataResource
from generator.diffusion.pipelines import Pipelines


def create_app(model_name, auth_token):
    if not torch.cuda.is_available():
        raise("CUDA not present. Quitting.")

    logging.debug(f"CUDA {torch.cuda.is_available()}")
    logging.debug(f"Torch version {torch.__version__}")

    pipelines = Pipelines(model_name)
    pipelines.preload_pipelines(auth_token)
    default_device = Device(pipelines)

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
