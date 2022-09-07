import torch
import sys
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast
import uuid
import logging
from config import Config
from pathlib import Path

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

        # Read prompt from command line
        if len(sys.argv) < 2:
            raise Exception("no text prompt was provided")

        prompt = clean(sys.argv[1])

        logging.debug(f"Torch version {torch.__version__}")
        id = uuid.uuid4()
        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")
        logging.info(f"prompt is [{prompt}]")

        outDir = Path(config_dict["generation"]["output_dir"])
        generate(config_dict["model"]["model_name"], prompt, f'{outDir.joinpath(filename)}', config_dict["model"]["guidance_scale"])

        logging.info(f"END generating {id}")
    except Exception as e:
        logging.exception(e)
        print(e, file=sys.stderr)
        print("FAIL")
    except:
        logging.exception("unhandled error occurred")
        print("FAIL")

# clean up the string - removing non utf-8 characters, check length
def clean(str):
    encoded = str.encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.strip()
    if len(cleaned) > 280:
        raise Exception("prompt must be less than 281 characters")
    return cleaned


# this does the actual image generation
def generate(model, prompt, filepath, guidance):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )

    pipe.to("cuda")  # Run on GPU

    with autocast():
        image = pipe(prompt, guidance_scale=guidance)["sample"][0]

    image.save(filepath)  # TODO write to a temp file and then change to output name


if __name__ == "__main__":
    try:
        config = Config()
        config.load()
        main(config)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except:
        print("Fatal error", file=sys.stderr)
        sys.exit(1)        