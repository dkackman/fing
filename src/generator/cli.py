import sys
import uuid
import logging
import torch
from config import Config
from pathlib import Path
from worker import generate

# TODO
# create lock on gpu

def main(config):
    try:
        config_dict=config.config_file
        logging.basicConfig(
            level=config_dict["generation"]["log_level"],
            filename=config.resolve_path(config_dict["generation"]["log_filename"]),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if not torch.cuda.is_available():
            raise Exception("unavailable")  # don't try to run this on cpu

        if len(sys.argv) < 2:
            raise Exception("no text prompt was provided")

        logging.debug(f"Torch version {torch.__version__}")

        # get the file name from the command line or generate one
        id = sys.argv[2] if len(sys.argv) > 2 else uuid.uuid4()

        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")

        image = generate(config_dict["model"]["model_name"], config_dict["model"]["guidance_scale"], sys.argv[1], config_dict["model"]["huggingface_token"])

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