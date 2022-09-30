from typing import List
from .device import Device
from threading import Lock

mutex: Lock = Lock()

available: List[Device] = []
busy: List[Device] = []


def add_device(device: Device):
    mutex.acquire(True, 2)
    try:
        available.append(device)
    finally:
        mutex.release()


def get_device() -> Device:
    mutex.acquire(True, 2)
    try:
        if len(available) > 0:
            device = available.pop()
            busy.append(device)
            return device

        raise (Exception("busy"))
    finally:
        mutex.release()


def release_device(device):
    mutex.acquire(True, 2)
    try:
        busy.remove(device)
        available.append(device)
    finally:
        mutex.release()
