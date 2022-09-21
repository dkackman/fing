from fastapi import APIRouter
from .. import info


info_router = APIRouter()


@info_router.get("/info")
def get_info():
    return info()
