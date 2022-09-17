# fing

A REST API for stable diffusion

## Introduction

This is a command line and REST interface to [stable-diffusion](https://github.com/CompVis/stable-diffusion). Used in part to generate art(?) and eventually make [Chia NFTs](https://www.chia.net/2022/06/29/1.4.0-introducing-the-chia-nft1-standard.en.html). ([rough example from with chia-repl](https://github.com/dkackman/chia-repl/blob/main/examples/scripts/txt2nft.js))

What you will need:

- A machine with a decent [CUDA](https://developer.nvidia.com/cuda-downloads) capable graphics card. (NVIDIA RTX 3060 or so - 8GB or more)
- [Anaconda](https://www.anaconda.com/) version 2022.05 (might work on older versions) on that machine
- [Node](https://nodejs.org/en/) on that machine
- I've done this all on ubuntu 22.04. Should work on other linuxes. Might even work on Windows.
- A [Huggingface account](https://huggingface.co/welcome) and [access token](https://huggingface.co/settings/tokens)
  - You will also need to accept [the model license agreement](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Config File

Create a directory `~/.fing/` and place a copy of [the config file](https://github.com/dkackman/fing/blob/main/src/config.yaml) there.
These settings need to be set specific to your environment:

- `output_dir` - Currently python and node communicate via the file system. This is the absolute path where the image will be generated and the node api will pick it up. Can be anywhere that both parts have access to.
- `huggingface_token` - The access token you got above. This is needed to use the model developed by _CompVis_ and licensed under the [OpenRAIL-M](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE) CreativeML license.

### Arguments

Both the CLI nd REST service share these argument names.

#### These arguments can be passed to do text to image

- `--prompt` - Required. The textual prompt to base the image on.
- `--num_images` - Defaults to 1. The number of images to create.
- `--guidance_scale` - Defaults to 7.5. The model guidance scale.
- `--num_inference_steps` - defaults to 50. The number of model inference steps.
- `--height` - defaults to 512. The image height.
- `--width` - defaults to 512. The image width.

#### To do image to image guided transformation the above (except height and width) and

- `--image_uri` - the URI of the init image
- `--strength` - The relative amount of noise to add to the init image.

#### To do in painting the above (excpet height and width) and

- `--mask_uri` - The URI of the mask image

### Command Line

Images are generated by a python program running in the conda environment.

**Important** - Make sure to run this python script at least once before trying the REST API. Not only will it validate that the config is correct, it will download the model checkpoints which are quite large, but only need to download once.

```bash
cd src
conda env create -f environment.yaml
conda activate fing

python generator/cli.py --prompt "An impressionist painting of penguin on a bicycle."
```

<img src="pb.jpg" width="200" height="200" alt="An impressionist painting of penguin on a bicycle."/>

### REST API

There is also a REST service interface:

```bash
cd src/generator
conda activate fing
python server.py
or 
gunicorn --bind 0.0.0.0:9147 server:gunicorn_app --timeout 120 
```

This will run [a simple REST api](https://dkackman.github.io/fing/) on port 9147:

<div>
http://localhost:3010/api/txt2img?prompt=Proof of space and time
</div>

<br>

<img src="post.jpg" width="256" height="256" alt="Proof of space and time."/>
