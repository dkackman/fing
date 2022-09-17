from flask_restful import reqparse, Resource
from threading import Lock
from urllib.parse import unquote

# TODO #5 create a worker (and mutex) per GPU
# there is only one of these per process right now
# this means only one job can be active at one time
mutex = Lock()

class SDResource(Resource):
    device = None

    def __init__(self, **kwargs):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, help="no prompt was provided", location='args', required=True, trim=True)
        parser.add_argument('guidance_scale', location='args', type=float, default=7.5)
        parser.add_argument('num_inference_steps', location='args', type=int, default=50)
        parser.add_argument('num_images', location='args', type=int, default=1)

        self.parser = parser
        self.device = kwargs["device"]


    # clean up the string - removing non utf-8 characters, check length
    def clean_prompt(str):
        encoded = unquote(str).encode("utf8", "ignore")
        decoded = encoded.decode("utf8", "ignore")  
        cleaned = decoded.replace('"' , "").replace("'", "").strip()
        if len(cleaned) > 280: # max length of a tweet
            raise Exception("prompt must be less than 281 characters")
            
        return cleaned