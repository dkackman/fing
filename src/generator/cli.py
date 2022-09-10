import sys
import uuid
import logging
import torch
from config import Config
from pathlib import Path
from worker import generate
import argparse
# TODO
# create lock on gpu

def main(config):
    config_dict=config.config_file
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        help="the prompt to render",
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=config_dict["generation"]["output_dir"]
    )    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        nargs="?",
        help="model guidance scale",
        default=7.5
    )
    parser.add_argument(
        "--num_images",
        type=int,
        nargs="?",        
        default=1,
        help="The number of images to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        nargs="?",        
        default=50,
        help="The number of inference steps",
    )    
    args = parser.parse_args()
    try:
        logging.basicConfig(
            level=config_dict["generation"]["log_level"],
            filename=config.resolve_path(config_dict["generation"]["log_filename"]),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        if not torch.cuda.is_available():
            raise Exception("unavailable")  # don't try to run this on cpu

        logging.debug(f"Torch version {torch.__version__}")

        # get the file name from the command line or generate one
        id = sys.argv[2] if len(sys.argv) > 2 else uuid.uuid4()

        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")

        prompt = args.prompt
        guidance_scale = args.guidance_scale
        num_inference_steps = args.num_inference_steps
        num_images = args.num_images
        image = generate(config_dict["model"]["model_name"], 
            guidance_scale,
            num_inference_steps, 
            num_images, 
            prompt, 
            config_dict["model"]["huggingface_token"]
        )

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