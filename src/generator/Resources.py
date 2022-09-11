from distutils.command.clean import clean
from flask import jsonify, send_file
from flask_restful import reqparse, abort, Resource
import logging
import worker
from urllib.parse import unquote
import io
from threading import Lock
import base64
import torch

# TODO #5 create a worker per GPU

mutex = Lock()

class txt2imgResource(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, help="no prompt was provided", location='args', required=True, trim=True)
        parser.add_argument('guidance_scale', location='args', type=float, default=7.5)
        parser.add_argument('num_inference_steps', location='args', type=int, default=50)
        parser.add_argument('num_images', location='args', type=int, default=1)
        parser.add_argument('height', location='args', type=int, default=512)
        parser.add_argument('width', location='args', type=int, default=512)

        self.parser = parser

    def get(self):
        args = self.parser.parse_args()

        try:
            prompt = clean_prompt(args.prompt)
            buffer = generate_image_buffer(
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                args.height,
                args.width,
                prompt
            )
            return send_file(buffer, mimetype="image/jpeg")
        except Exception as e:
            print(e)
            abort(500)


class txt2imgMetadataResource(txt2imgResource):
    def __init__(self):
        super(txt2imgMetadataResource, self).__init__()

    def get(self):
        try:
            args = self.parser.parse_args()

            prompt = clean_prompt(args.prompt)
            buffer = generate_image_buffer(
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                args.height,
                args.width,
                prompt
            )
            metadata = info()
            metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
            metadata["parameters"] = {
                'guidance_scale': args.guidance_scale,
                'num_inference_steps': args.num_inference_steps,
                'num_images': args.num_images,
                'height': args.height,
                'width': args.width,
                'prompt': prompt
                }
            return jsonify(metadata)   

        except Exception as e:
            print(e)
            abort(500)


def generate_image_buffer(guidance_scale, num_inference_steps, num_images, height, width, prompt):
    try:
        # only allow one image generation at a time        
        locked = mutex.acquire(False)
        if locked:
            logging.info(f"START generating")
            image = worker.generate( 
                guidance_scale, 
                num_inference_steps, 
                num_images, 
                height, 
                width, 
                prompt
            )
            logging.info(f"END generating")
        else:
            abort(423, "Busy. Try again later.")
    finally:
        if locked:
            mutex.release()

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer

class InfoResource(Resource):
    def get(self):
        return jsonify(info())

def info():
    return {
            'software': {
                'name': "fing",
                'version': "0.1.0",
                'torch_version': torch.__version__,
            },
            'model': worker.pipe.config
        }


# clean up the string - removing non utf-8 characters, check length
def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")  
    cleaned = decoded.replace('"' , "").replace("'", "").strip()
    if len(cleaned) > 280: # max length of a tweet
        raise Exception("prompt must be less than 281 characters")
        
    return cleaned
