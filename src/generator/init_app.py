from .settings import Settings, settings_exist, save_settings, get_settings_full_path
import sys
import json


def init():
    if settings_exist():
        print("Config already set. Nothing to do.")
        return

    settings = Settings()

    print("Provide the following details for the intial configuration:\n")
    token = input("Huggingface API token: ").strip()
    if len(token) == 0:
        print("A Huggingface API token is required.")
        return

    model_cache_dir = input("Model cache directory (/tmp): ").strip()
    model_cache_dir = "/tmp" if len(model_cache_dir) == 0 else model_cache_dir

    host = input("Service host (localhost): ").strip()
    host = "localhost" if len(host) == 0 else host

    port = input("Service port (9147): ").strip()
    port = 9147 if len(port) == 0 else int(port)

    settings.huggingface_token = token
    settings.model_cache_dir = model_cache_dir
    settings.host = host
    settings.port = port

    print("\n")
    print(settings.json(indent=2))

    confirm = input("Is this corrent? (Y/n): ").strip().lower()
    if len(confirm) == 0 or confirm.startswith("y"):
        save_settings(settings)
        print(f"Configuraiton saved to {get_settings_full_path()}")
    else:
        print("Cancelled")
