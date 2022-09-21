from fastapi import APIRouter, Depends
from .. import info
from .x_api_key import x_api_key_auth

info_router = APIRouter()


@info_router.get("/info", dependencies=[Depends(x_api_key_auth)])
def get_info():
    return info()
