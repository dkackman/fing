from flask import jsonify
from flask_restful import Resource
from .. import info

class InfoResource(Resource):
    def get(self):
        return jsonify(info())
