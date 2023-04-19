# fing

A REST API for stable diffusion

(though I've stopped working on this and now spend more time [over here](https://github.com/dkackman/chiaSWARM)

## Introduction

This is a REST interface to [stable-diffusion](https://github.com/CompVis/stable-diffusion).

What you will need:

- A machine with a decent [CUDA](https://developer.nvidia.com/cuda-downloads) capable graphics card. (NVIDIA GTX 1080, RTX 3060 or so - 8GB or more)
- [Anaconda](https://www.anaconda.com/) version 2022.05 (might work on older versions) on that machine
- I've done this all on ubuntu 22.04. Should work on other linuxes.
- A [Huggingface account](https://huggingface.co/welcome) and [access token](https://huggingface.co/settings/tokens)
  - You will also need to accept [the model license agreement](https://huggingface.co/CompVis/stable-diffusion-v1-4)
  
## Features

- Includes txt2img, img2img, imginpaint, and face generation
- Pipeline cache so any of the above can be swapped into the GPU as needed
- GPU pooling to support multiple concurrent requests (1 per GPU)
- Optional `x-api-key` for simple auth
- [Open API swagger spec](https://dkackman.github.io/fing/)
  - once the server is running the latest spec can always be viewed at `<server_address:port>/docs`

### Setup

```bash
cd src
conda env create -f environment.yaml
conda activate fing
python -m generator.init_app
```

This will create `settings.json` file in `~/.fing` as well as download all of the diffuser model caches. This last part can take some time.

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
