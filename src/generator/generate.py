import torch
import sys
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast
import uuid
import logging

# TODO - load values from config.yaml


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        filename="app.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Read prompt from command line
    if len(sys.argv) < 2:
        raise Exception("no text prompt was provided")

    prompt = clean(sys.argv[1])

    if not torch.cuda.is_available():
        raise Exception("unavailable")  # don't try to run this on cpu

    id = uuid.uuid4()
    filename = f"{id}.jpg"
    print(filename)  # this let's the caller know what file to look for

    logging.info(f"START generating {id}")
    logging.info(f"prompt is [{prompt}]")

    generate("CompVis/stable-diffusion-v1-4", prompt, f"/mnt/ml/outputs/{filename}")

    logging.info(f"END generating {id}")


# clean up the string - removing non utf-8 characters, check length
def clean(str):
    encoded = str.encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.strip()
    if len(cleaned) > 280:
        raise Exception("prompt must be less than 281 characters")
    return cleaned


# this does the actual image generation
def generate(model, prompt, filepath):
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True,
    )

    pipe.to("cuda")  # Run on GPU

    with autocast():
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]

    image.save(filepath)  # TODO write to a temp file and then change to output name


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        print(e, file=sys.stderr)
        print("FAIL")
    except:
        logging.exception("unhandled error occurred")
        print("FAIL")
