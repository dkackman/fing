import sys
from flask import Flask, jsonify, send_file
from flask_restful import reqparse, abort, Api, Resource
from config import Config
import logging
import torch
import worker
from urllib.parse import unquote
import io

# TODO 
# - run in WSGI server
# - SSL
# - auth
# - mutex on gpu busy
# - manage multiple gpus

parser = reqparse.RequestParser()
parser.add_argument('prompt', type=str, help="no prompt was provided", location='args')

config = Config().load()

class InfoResource(Resource):
    def get(self):
        info = {
            'version': "0.1.0",
            'torch': torch.__version__
        }
        return jsonify(info)

class ImageResource(Resource):
    def get(self):
        config_dict = config.config_file
        args = parser.parse_args()
        prompt = args["prompt"]
        if prompt is None:
            abort(400, message="prompt argument not found")

        try:
            logging.info(f"START generating")
            image = worker.generate_with_pipe(pipe, config_dict["model"]["guidance_scale"], unquote(prompt))
            logging.info(f"END generating")

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            return send_file(buffer, mimetype="image/jpeg")

        except Exception as e:
            print(e)
            abort(500)


def main():
    config_dict = config.config_file
    logging.basicConfig(
        level=config_dict["generation"]["log_level"],
        filename=config.resolve_path(config_dict["generation"]["log_filename"]),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not torch.cuda.is_available():
        logging.critical("CUDA not available")
        raise Exception("unavailable")  # don't try to run this on cpu
    
    logging.info(f'Starting server at {config_dict["generation"]["host"]}:{config_dict["generation"]["port"]}')
    logging.debug(f"Torch version {torch.__version__}")

    # load the model into the gpu - stays there for the life of the process
    global pipe 
    pipe = worker.load_model(config_dict["model"]["model_name"], config_dict["model"]["huggingface_token"])

    app = Flask("text2img service")
    api = Api(app)

    api.add_resource(InfoResource, '/')
    api.add_resource(ImageResource, '/generate')
    print(config_dict["generation"]["port"])
    app.run()
    #app.run(config_dict["generation"]["host"], config_dict["generation"]["port"])
    logging.info("Server exiting")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except:
        print("Fatal error", file=sys.stderr)
        sys.exit(1)        