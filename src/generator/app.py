from flask import Flask, jsonify, send_file
from flask_restful import reqparse, abort, Api, Resource
import logging
import torch
import worker
from urllib.parse import unquote
import io
from threading import Lock

parser = reqparse.RequestParser()
parser.add_argument('prompt', type=str, help="no prompt was provided", location='args', required=True, trim=True)
parser.add_argument('guidance_scale', location='args', type=float, default=7.5)
parser.add_argument('num_inference_steps', location='args', type=int, default=50)
parser.add_argument('num_images', location='args', type=int, default=1)
parser.add_argument('height', location='args', type=int, default=512)
parser.add_argument('width', location='args', type=int, default=512)

mutex = Lock()

def create_app(model_name, auth_token):
    if not torch.cuda.is_available():
        logging.critical("CUDA not available")
        raise Exception("unavailable")  # don't try to run this on cpu
    
    logging.debug(f"Torch version {torch.__version__}")

    # load the model into the gpu - stays there for the life of the process
    global pipe 
    pipe = worker.load_model(model_name, auth_token)

    app = Flask("txt2img service")
    api = Api(app)

    api.add_resource(InfoResource, '/')
    api.add_resource(txt2imgResource, '/txt2img')

    return app


class txt2imgResource(Resource):
    def get(self):
        args = parser.parse_args()

        try:
            # only allow one image generation at a time
            # TODO create a worker per GPU
            locked = mutex.acquire(False)
            if locked:
                return do_work(
                    args.guidance_scale,
                    args.num_inference_steps, 
                    args.num_images, 
                    args.height,
                    args.width,
                    args.prompt
                )
            
            abort(423, "Busy. Try again later.")

        finally:
            if locked:
                mutex.release()


def do_work(guidance_scale, num_inference_steps, num_images, height, width, prompt):
    try:
        logging.info(f"START generating")
        image = worker.generate_with_pipe(pipe, 
            guidance_scale, 
            num_inference_steps, 
            num_images, 
            height, 
            width, 
            unquote(prompt)
        )
        logging.info(f"END generating")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        return send_file(buffer, mimetype="image/jpeg")

    except Exception as e:
        print(e)
        abort(500)


class InfoResource(Resource):
    def get(self):
        info = {
            'version': "0.1.0",
            'torch': torch.__version__
        }
        return jsonify(info)