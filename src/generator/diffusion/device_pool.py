from .device import Device

device_list = []


def add_device(device: Device):
    device_list.append(device)


def get_device() -> Device:
    # our pool can only have one deivce right now
    return device_list[0]