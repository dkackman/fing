import sys
import uuid
import logging
import torch
import argparse
import torch
from pathlib import Path
from . import setup_logging
from .external_resource import get_image
from .config import Config
from .diffusion.pipelines import Pipelines
from .diffusion.device import Device
from .init_config import init


if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")


def main(config):
    config_dict = config.config_file
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, nargs="?", help="the prompt to render", required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=".",
    )
    parser.add_argument(
        "--output_name", type=str, nargs="?", help="The name of the output file"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        nargs="?",
        help="model guidance scale",
        default=7.5,
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs="?",
        help="0-1 indicates how much to transform the reference (img2img only)",
        default=0.75,
    )
    parser.add_argument(
        "--image_uri",
        type=str,
        nargs="?",
        help="Uri of an image to transform - if provided triggers img2ing  or imginpaint instead of text2img",
    )
    parser.add_argument(
        "--mask_uri",
        type=str,
        nargs="?",
        help="Uri of a mask image to use for imginpaint - if provided with --image_uri, triggers imginpaint instead of img2ing",
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
    parser.add_argument(
        "--height",
        type=int,
        nargs="?",
        default=512,
        help="The image height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        nargs="?",
        default=512,
        help="The image width in pixels",
    )
    parser.add_argument(
        "--dont_conserve_memory",
        action=argparse.BooleanOptionalAction,
        help="Don't use revision fp16 or enable_attention_slicing",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Verbose output",
    )
    args = parser.parse_args()
    try:
        setup_logging(config)

        logging.debug(f"Torch version {torch.__version__}")

        # get the file name from the command line or generate one
        id = args.output_name if args.output_name is not None else uuid.uuid4()

        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")

        auth_token = config_dict["model"]["huggingface_token"]
        pipelines = Pipelines(
            config_dict["model"]["model_name"],
            config_dict["generation"]["model_cache_dir"],
        )

        prompt = args.prompt.replace('"', "").replace("'", "")
        if args.image_uri is not None:
            if args.mask_uri is not None:
                pipelines.preload_pipelines(
                    auth_token, ["imginpaint"], not args.dont_conserve_memory
                )
                device = Device(pipelines)
                init_image = get_image(args.image_uri)
                mask_image = get_image(args.mask_uri)

                image, pipe_config = device(
                    pipeline_name="imginpaint",
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images=args.num_images,
                    prompt=prompt,
                    init_image=init_image,
                    mask_image=mask_image,
                )
            else:
                pipelines.preload_pipelines(
                    auth_token, ["img2img"], not args.dont_conserve_memory
                )
                device = Device(pipelines)
                init_image = get_image(args.image_uri)

                image, pipe_config = device(
                    pipeline_name="img2img",
                    strength=args.strength,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images=args.num_images,
                    prompt=prompt,
                    init_image=init_image,
                )
        else:
            pipelines.preload_pipelines(
                auth_token, ["txt2img"], not args.dont_conserve_memory
            )
            device = Device(pipelines)

            image, pipe_config = device(
                pipeline_name="txt2img",
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images=args.num_images,
                height=args.height,
                width=args.width,
                prompt=prompt,
            )

        outDir = Path(args.outdir)
        image.save(f"{outDir.joinpath(filename)}")
        if args.verbose:
            print(pipe_config)

        logging.info(f"END generating {id}")
    except Exception as e:
        logging.exception(e)
        print(e, file=sys.stderr)
        raise e
    except:
        logging.exception("Unhandled error occurred")
        raise Exception("FAIL")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "init":
        init()
    else:
        main(Config().load())
