from flask import jsonify, send_file
from flask_restful import reqparse, abort, Resource
import base64
from web_worker import clean_prompt, generate_txt2img_buffer, info


class img2imgResource(Resource):
    model = None

    def __init__(self, **kwargs):
        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, help="no prompt was provided", location='args', required=True, trim=True)
        parser.add_argument('guidance_scale', location='args', type=float, default=7.5)
        parser.add_argument('num_inference_steps', location='args', type=int, default=50)
        parser.add_argument('num_images', location='args', type=int, default=1)

        self.parser = parser
        self.model = kwargs["model"]


    def get(self):
        args = self.parser.parse_args()

        try:
            prompt = clean_prompt(args.prompt)
            buffer = generate_txt2img_buffer(
                self.model,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                prompt
            )
            return send_file(buffer, mimetype="image/jpeg")
        except Exception as e:
            print(e)
            abort(500)


class img2imgMetadataResource(img2imgResource):
    def __init__(self, **kwargs):
        super(img2imgResource, self).__init__(**kwargs)

    def get(self):
        try:
            args = self.parser.parse_args()

            prompt = clean_prompt(args.prompt)
            buffer = generate_txt2img_buffer(
                self.model,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                args.height,
                args.width,
                prompt
            )
            metadata = info(self.model)
            metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
            metadata["parameters"] = {
                'guidance_scale': args.guidance_scale,
                'num_inference_steps': args.num_inference_steps,
                'num_images': args.num_images,
                'prompt': prompt
                }
            return jsonify(metadata)   

        except Exception as e:
            print(e)
            abort(500)
