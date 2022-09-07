import sys
import uuid
import logging
import torch
from config import Config
from pathlib import Path
from worker import generate

# TODO
# create RPC service
# create lock on gpu

def main(config):
    try:
        config_dict=config.config_file
        logging.basicConfig(
            level=config_dict["logging"]["log_level"],
            filename=config.resolve_path(config_dict["logging"]["log_filename"]),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if not torch.cuda.is_available():
            raise Exception("unavailable")  # don't try to run this on cpu

        if len(sys.argv) < 2:
            raise Exception("no text prompt was provided")

        logging.debug(f"Torch version {torch.__version__}")
        id = uuid.uuid4()
        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")

        image = generate(config_dict["model"]["model_name"], config_dict["model"]["guidance_scale"], sys.argv[1])

        outDir = Path(config_dict["generation"]["output_dir"])
        image.save(f'{outDir.joinpath(filename)}')  # TODO write to a temp file and then change to output name

        logging.info(f"END generating {id}")
    except Exception as e:
        logging.exception(e)
        print(e, file=sys.stderr)
        print("FAIL")
    except:
        logging.exception("Unhandled error occurred")
        print("FAIL")


if __name__ == "__main__":
    try:
        main(Config().load())
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except:
        print("Fatal error", file=sys.stderr)
        sys.exit(1)        