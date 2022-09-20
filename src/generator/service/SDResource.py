from flask import send_file
from flask_restful import reqparse, Resource, abort
from urllib.parse import unquote
import logging
import io
from .x_api import api_key_required


class SDResource(Resource):
    device = None

    def __init__(self, **kwargs):
        parser = reqparse.RequestParser()
        parser.add_argument(
            "prompt",
            type=str,
            help="no prompt was provided",
            location="args",
            required=True,
            trim=True,
        )
        parser.add_argument("guidance_scale", location="args", type=float, default=7.5)
        parser.add_argument(
            "num_inference_steps", location="args", type=int, default=50
        )
        parser.add_argument("num_images", location="args", type=int, default=1)

        self.parser = parser
        self.device = kwargs["device"]

    @api_key_required
    def get(self):
        args = self.parser.parse_args()

        try:
            buffer, pipe_config = self.generate_buffer(args)
            return send_file(buffer, mimetype="image/jpeg")
        except Exception as e:
            print(e)
            abort(500)

    def generate_buffer(self, **kwargs):
        try:
            logging.info(f"START generating {kwargs['pipeline_name']}")

            kwargs["prompt"] = clean_prompt(kwargs["prompt"])
            image, pipe_config = self.device(**kwargs)

            logging.info(f"END generating {kwargs['pipeline_name']}")
        except:
            abort(423)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        return buffer, pipe_config


# clean up the string - removing non utf-8 characters, check length
def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()
    if len(cleaned) > 280:  # max length of a tweet
        raise Exception("prompt must be less than 281 characters")

    return cleaned
