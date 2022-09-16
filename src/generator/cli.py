import sys
import uuid
import logging
import torch
from config import Config
from pathlib import Path
from gpu import Gpu
import argparse
from log_setup import setup_logging
import torch
from external_resource import get_image
from pipelines import Pipelines

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
        "--output_name",
        type=str,
        nargs="?",
        help="The name of the output file"
    )        
    parser.add_argument(
        "--guidance_scale",
        type=float,
        nargs="?",
        help="model guidance scale",
        default=7.5
    )
    parser.add_argument(
        "--strength",
        type=float,
        nargs="?",
        help="0-1 indicates how much to transform the reference (img2img only)",
        default=0.75
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
    args = parser.parse_args()
    try:
        setup_logging(config)

        if not torch.cuda.is_available():
            raise Exception("unavailable")  # don't try to run this on cpu

        logging.debug(f"Torch version {torch.__version__}")

        # get the file name from the command line or generate one
        id = args.output_name if args.output_name is not None else uuid.uuid4()

        filename = f"{id}.jpg"
        print(filename)  # this let's the caller know what file to look for

        logging.info(f"START generating {id}")

        auth_token = config_dict["model"]["huggingface_token"]
        pipelines = Pipelines(config_dict["model"]["model_name"])

        prompt = args.prompt.replace('"' , "").replace("'", "")
        if args.image_uri is not None:
            if args.mask_uri is not None:
                pipelines.preload_pipelines(auth_token, ["imginpaint"])
                gpu = Gpu(pipelines)
                init_image = get_image(args.image_uri)
                mask_image = get_image(args.mask_uri)

                image, pipe_config = gpu.get_imginpaint(
                    args.strength,
                    args.guidance_scale,
                    args.num_inference_steps, 
                    args.num_images, 
                    prompt,
                    init_image,
                    mask_image
                )
            else:
                pipelines.preload_pipelines(auth_token, ["img2img"])
                gpu = Gpu(pipelines)
                init_image = get_image(args.image_uri)

                image, pipe_config = gpu.get_img2img(
                    args.strength,
                    args.guidance_scale,
                    args.num_inference_steps, 
                    args.num_images, 
                    prompt,
                    init_image
                )
        else:
            pipelines.preload_pipelines(auth_token, ["txt2img"])            
            gpu = Gpu(pipelines)
            image, pipe_config= gpu.get_txt2img(
                args.guidance_scale,
                args.num_inference_steps, 
                args.num_images, 
                args.height,
                args.width,
                prompt
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