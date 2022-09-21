from fastapi import APIRouter, Depends
from .. import info, InfoModel
from .x_api_key import x_api_key_auth

info_router = APIRouter()


@info_router.get(
    "/info",
    dependencies=[Depends(x_api_key_auth)],
    tags=["Service Information"],
    response_model=InfoModel,
)
def get_info() -> InfoModel:
    return info()
