from flask import jsonify, send_file
from flask_restful import reqparse, abort, Resource
import base64
from generator.service.web_worker import clean_prompt, generate_img2img_buffer
from generator.external_resource import get_image
from .. import info


class img2imgResource(Resource):
    device = None

    def __init__(self, **kwargs):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, help="no prompt was provided", location='args', required=True, trim=True)
        parser.add_argument('image_uri', type=str, help="no image uri was provided", location='args', required=True, trim=True)
        parser.add_argument('strength', location='args', type=float, default=0.75)
        parser.add_argument('guidance_scale', location='args', type=float, default=7.5)
        parser.add_argument('num_inference_steps', location='args', type=int, default=50)
        parser.add_argument('num_images', location='args', type=int, default=1)

        self.parser = parser
        self.device = kwargs["device"]


    def get(self):
        args = self.parser.parse_args()

        try:
            init_image = get_image(args.image_uri)
            prompt = clean_prompt(args.prompt)
            buffer, pipe_config = generate_img2img_buffer(
                self.device,
                args.strength,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                prompt,
                init_image
            )
            return send_file(buffer, mimetype="image/jpeg")
        except Exception as e:
            print(e)
            abort(500)


class img2imgMetadataResource(img2imgResource):
    def __init__(self, **kwargs):
        super(img2imgMetadataResource, self).__init__(**kwargs)


    def get(self):
        try:
            args = self.parser.parse_args()

            init_image = get_image(args.image_uri)
            prompt = clean_prompt(args.prompt)
            buffer, pipe_config = generate_img2img_buffer(
                self.device,
                args.strength,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                prompt,
                init_image
            )
            metadata = info()
            metadata["pipe_config"] = pipe_config
            metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
            metadata["parameters"] = {
                'guidance_scale': args.guidance_scale,
                'num_inference_steps': args.num_inference_steps,
                'num_images': args.num_images,
                'prompt': prompt,
                'strength': args.strength,
                'image_uri': args.image_uri
            }
            return jsonify(metadata)   

        except Exception as e:
            print(e)
            abort(500)
