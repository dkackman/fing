from flask import request, jsonify
from functools import wraps
from collections import namedtuple


x_api_config = namedtuple("x_api_config", ("enabled", "key_list"))
x_config = x_api_config(False, [])


def enable_x_api_enforcement(key_list):
    if key_list is None or not isinstance(key_list, list) or len(key_list) == 0:
        raise ("A list of valid api keys must be provided")

    global x_config
    x_config = x_api_config(True, key_list)


def api_key_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        if x_config.enabled:
            key = None

            if "x-api-key" in request.headers:
                key = request.headers["x-api-key"]

            if not key:
                return jsonify({"message": "a valid api key is missing"})

            if key not in x_config.key_list:
                return jsonify({"message": "api key is invalid"})

        return f(*args, **kwargs)

    return decorator
