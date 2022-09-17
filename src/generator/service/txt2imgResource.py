from flask import jsonify
from flask_restful import abort
import base64
from .SDResource import SDResource
from .. import info


class txt2imgResource(SDResource):

    def __init__(self, **kwargs):
        super(txt2imgResource, self).__init__(**kwargs)
        self.parser.add_argument('height', location='args', type=int, default=512)
        self.parser.add_argument('width', location='args', type=int, default=512)
    
    
    def generate_buffer(self, args):
        return super().generate_buffer(
                        pipeline_name="txt2img",
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps, 
                        num_images=args.num_images, 
                        height=args.height,
                        width=args.width,
                        prompt=args.prompt
                    )


class txt2imgMetadataResource(txt2imgResource):
    def __init__(self, **kwargs):
        super(txt2imgMetadataResource, self).__init__(**kwargs)


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
