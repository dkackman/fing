from fastapi import APIRouter, Depends
from .x_api_key import x_api_key_auth
from ..diffusion.device_pool import get_available_gpu_count
import torch

status_router = APIRouter()


@status_router.get(
    "/status",
    dependencies=[Depends(x_api_key_auth)],
    tags=["Service Information"],
)
def get_stats():
    return {
        "gpu_count": torch.cuda.device_count(),
        "available_gpu_count": get_available_gpu_count(),
    }
