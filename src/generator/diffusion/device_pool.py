from typing import List
from .device import Device
from threading import Lock

mutex: Lock = Lock()

available: List[Device] = []

# TODO - avoid loading if possible by finding a device with the desired pipeline already on it


def add_device_to_pool(device: Device):
    mutex.acquire(True, 2)
    try:
        available.append(device)
    finally:
        mutex.release()


def remove_device_from_pool() -> Device:
    mutex.acquire(True, 2)
    try:
        if len(available) > 0:
            return available.pop(0)

        raise Exception("busy")
    finally:
        mutex.release()
