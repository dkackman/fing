from .config import Config
from pathlib import Path
import os
import yaml
import sys


def init():
    if Config.exists():
        print("Config already set. Nothing to do.")
        return

    this_dir = Path(os.path.realpath(__file__)).parent
    config = Config.load_from(this_dir.joinpath("config.yaml"))

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

    config["model"]["huggingface_token"] = token
    config["generation"]["model_cache_dir"] = model_cache_dir
    config["generation"]["host"] = host
    config["generation"]["port"] = port

    print("\n")
    yaml.dump(config, sys.stdout)

    confirm = input("Is this corrent? (Y/n): ").strip().lower()
    if len(confirm) == 0 or confirm.startswith("y"):
        Config.save_config(config)
        print(f"Configuraiton saved to {Config.get_full_path()}")
    else:
        print("Cancelled")
