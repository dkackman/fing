from flask import jsonify, send_file
from flask_restful import abort
import base64
import logging
import io
from .SDResource import SDResource, mutex
from .. import info


class txt2imgResource(SDResource):

    def __init__(self, **kwargs):
        super(txt2imgResource, self).__init__(**kwargs)
        self.parser.add_argument('height', location='args', type=int, default=512)
        self.parser.add_argument('width', location='args', type=int, default=512)


    def get(self):
        args = self.parser.parse_args()

        try:
            prompt = SDResource.clean_prompt(args.prompt)
            buffer, pipe_config = self.generate_txt2img_buffer(
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


    def generate_txt2img_buffer(self, guidance_scale, num_inference_steps, num_images, height, width, prompt):
        try:
            # only allow one image generation at a time        
            locked = mutex.acquire(False)
            if locked:
                logging.info(f"START txt2img generating")

                image, config = self.device.get_txt2img( 
                    guidance_scale, 
                    num_inference_steps, 
                    num_images, 
                    height, 
                    width, 
                    prompt
                )
                logging.info(f"END txt2img generating")
            else:
                abort(423, "Busy. Try again later.")
        finally:
            if locked:
                mutex.release()

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        return buffer, config


class txt2imgMetadataResource(txt2imgResource):
    def __init__(self, **kwargs):
        super(txt2imgMetadataResource, self).__init__(**kwargs)


    def get(self):
        try:
            args = self.parser.parse_args()

            prompt = SDResource.clean_prompt(args.prompt)
            buffer, pipe_config = self.generate_txt2img_buffer(
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                args.height,
                args.width,
                prompt
            )
            metadata = info()
            metadata["pipe_config"] = pipe_config            
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
