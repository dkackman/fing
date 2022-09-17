from flask import jsonify, send_file
from flask_restful import abort
import base64
import logging
import io
from .SDResource import SDResource, mutex
from .. import info
from ..external_resource import get_image


class imginpaintResource(SDResource):

    def __init__(self, **kwargs):
        super(imginpaintResource, self).__init__(**kwargs)

        self.parser.add_argument('image_uri', type=str, help="no image uri was provided", location='args', required=True, trim=True)
        self.parser.add_argument('mask_uri', type=str, help="no mask uri was provided", location='args', required=True, trim=True)
        self.parser.add_argument('strength', location='args', type=float, default=0.75)


    def get(self):
        args = self.parser.parse_args()

        try:
            init_image = get_image(args.image_uri)
            mask_image = get_image(args.mask_uri)
            prompt = SDResource.clean_prompt(args.prompt)
            buffer, pipe_config = self.generate_imginpaint_buffer(
                args.strength,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                prompt,
                init_image,
                mask_image
            )
            return send_file(buffer, mimetype="image/jpeg")
        except Exception as e:
            print(e)
            abort(500)


    def generate_imginpaint_buffer(self, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image, mask_image):
        try:
            # only allow one image generation at a time        
            locked = mutex.acquire(False)
            if locked:
                logging.info(f"START img2img generating")

                image, config = self.device.get_imginpaint( 
                    strength,
                    guidance_scale, 
                    num_inference_steps, 
                    num_images, 
                    prompt,
                    init_image,
                    mask_image
                )
                logging.info(f"END img2img generating")
            else:
                abort(423, "Busy. Try again later.")
        finally:
            if locked:
                mutex.release()

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        return buffer, config


class imginpaintMetadataResource(imginpaintResource):
    def __init__(self, **kwargs):
        super(imginpaintMetadataResource, self).__init__(**kwargs)


    def get(self):
        try:
            args = self.parser.parse_args()

            init_image = get_image(args.image_uri)
            mask_image = get_image(args.mask_uri)
            prompt = SDResource.clean_prompt(args.prompt)
            buffer, pipe_config = self.generate_imginpaint_buffer(
                args.strength,
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                prompt,
                init_image,
                mask_image
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
                'image_uri': args.image_uri,
                'mask_uri': args.mask_uri
            }
            return jsonify(metadata)   

        except Exception as e:
            print(e)
            abort(500)
