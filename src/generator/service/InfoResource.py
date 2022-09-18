from flask import jsonify
from flask_restful import Resource
from .. import info
from .x_api import api_key_required


class InfoResource(Resource):
    @api_key_required
    def get(self):
        return jsonify(info())
