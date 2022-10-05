# fing

A REST API for stable diffusion

## Introduction

This is a command line and REST interface to [stable-diffusion](https://github.com/CompVis/stable-diffusion). Used in part to generate art(?) and eventually make [Chia NFTs](https://www.chia.net/2022/06/29/1.4.0-introducing-the-chia-nft1-standard.en.html). ([rough example from with chia-repl](https://github.com/dkackman/chia-repl/blob/main/examples/scripts/txt2nft.js))

What you will need:

- A machine with a decent [CUDA](https://developer.nvidia.com/cuda-downloads) capable graphics card. (NVIDIA RTX 3060 or so - 8GB or more)
- [Anaconda](https://www.anaconda.com/) version 2022.05 (might work on older versions) on that machine
- [Node](https://nodejs.org/en/) on that machine
- I've done this all on ubuntu 22.04. Should work on other linuxes.
- A [Huggingface account](https://huggingface.co/welcome) and [access token](https://huggingface.co/settings/tokens)
  - You will also need to accept [the model license agreement](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Setup

```bash
cd src
conda env create -f environment.yaml
python -m generator.init_app
```

This will create config file in `~/.fing` as well as download all of the diffuser model caches. This can take some time.

### Arguments

The REST service has argument names.

#### These arguments can be passed to do text to image

- `prompt` - Required. The textual prompt to base the image on.
- `num_images` - Defaults to 1. The number of images to create.
- `guidance_scale` - Defaults to 7.5. The model guidance scale.
- `num_inference_steps` - defaults to 50. The number of model inference steps.
- `height` - defaults to 512. The image height.
- `width` - defaults to 512. The image width.

#### To do image to image guided transformation, use the above (except height and width) and

- `image_uri` - the URI of the init image
- `strength` - The relative amount of noise to add to the init image.

#### To do in painting, use the above (except height and width) and

- `mask_uri` - The URI of the mask image

### Running the Service

The rest service can be started this way:

```bash
cd src
conda activate fing

python -m generator.server
or 
uvicorn generator.server:app --host 0.0.0.0 --port 9147
```

This will run [a simple REST api](https://dkackman.github.io/fing/) on port 9147:

<div>
http://localhost:3010/api/txt2img?prompt=Proof of space and time
</div>

<br>

<img src="post.jpg" width="256" height="256" alt="Proof of space and time."/>
