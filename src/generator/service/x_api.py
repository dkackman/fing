from flask import request, jsonify  
from functools import wraps


enforce = False

def api_key_required(f):  
    @wraps(f)  
    def decorator(*args, **kwargs):
        key = None 

        if 'x-api-key' in request.headers:  
            key = request.headers['x-api-key'] 

        if not key:  
            return jsonify({'message': 'a valid api key is missing'})   

        try:  
            validate_key(key)
        except:  
            return jsonify({'message': 'api key is invalid'})  
  
        return f(*args,  **kwargs) 
        
    return decorator 

def validate_key(key):
    if enforce:
        raise("invalid key")

