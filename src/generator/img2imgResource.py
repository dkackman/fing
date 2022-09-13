from flask import jsonify, send_file
from flask_restful import reqparse, abort, Resource
import base64
from web_worker import clean_prompt, generate_image_buffer, info


class img2imgResource(Resource):
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


class img2imgMetadataResource(img2imgResource):
    def __init__(self):
        super(img2imgResource, self).__init__()

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
