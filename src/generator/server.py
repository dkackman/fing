from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource

app = Flask("text2img service")
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('prompt')

class ImageResource(Resource):
    def get(self):
        args = parser.parse_args()
        return jsonify({'square': args})


api.add_resource(ImageResource, '/generate')

if __name__ == '__main__':
    app.run()