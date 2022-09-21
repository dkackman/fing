from collections import namedtuple
from fastapi.security import APIKeyHeader
from fastapi.exceptions import HTTPException
from fastapi import Depends

x_api_config = namedtuple("x_api_config", ("enabled", "key_list"))
x_config = x_api_config(False, [])

X_API_KEY = APIKeyHeader(name="X-API-Key")


def enable_x_api_keys(key_list):
    if key_list is None or not isinstance(key_list, list) or len(key_list) == 0:
        raise Exception("A list of valid api keys must be provided")

    global x_config
    x_config = x_api_config(True, key_list)


def x_api_key_auth(x_api_key: str = Depends(X_API_KEY)):
    if x_config.enabled:
        if x_api_key not in x_config.key_list:
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
            )
