from flask import jsonify
from flask_restful import abort
import base64
from .SDResource import SDResource
from .. import info
from ..external_resource import get_image


class img2imgResource(SDResource):

    def __init__(self, **kwargs):
        super(img2imgResource, self).__init__(**kwargs)

        self.parser.add_argument('image_uri', type=str, help="no image uri was provided", location='args', required=True, trim=True)
        self.parser.add_argument('strength', location='args', type=float, default=0.75)


    def generate_buffer(self, args):
        return super().generate_buffer(
                        pipeline_name="img2img",
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps, 
                        num_images=args.num_images, 
                        prompt=args.prompt,
                        init_image=get_image(args.image_uri)
                    )


class img2imgMetadataResource(img2imgResource):

    def __init__(self, **kwargs):
        super(img2imgMetadataResource, self).__init__(**kwargs)


    def get(self):
        args = self.parser.parse_args()
        try:
            buffer, pipe_config = self.generate_buffer(args)
            metadata = info()
            metadata["pipe_config"] = pipe_config
            metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
            metadata["parameters"] = args
            return jsonify(metadata)   

        except Exception as e:
            print(e)
            abort(500)
