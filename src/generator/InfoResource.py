from flask import jsonify
from flask_restful import Resource
from web_worker import info

class InfoResource(Resource):
    def get(self):
        return jsonify(info())
