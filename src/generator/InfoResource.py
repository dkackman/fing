from flask import jsonify
from flask_restful import Resource
from web_worker import info

class InfoResource(Resource):
    model = None

    def __init__(self, **kwargs):
        self.model = kwargs["model"]


    def get(self):
        return jsonify(info(self.model))
