# Worker mode

## Install

### Resources

- [Install nvidia drivers ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [NVIDIA Docker Overview](https://hub.docker.com/r/nvidia/cuda)

### Prepare the Environment

#### Ubuntu 22.10

```bash
# install linux nvidia drivers
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall # reboot after
sudo apt install nvtop

# install miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh
conda update conda
```

#### Windows

```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o Miniconda3-latest-Windows-x86_64.exe
./Miniconda3-latest-Windows-x86_64.exe
```

### Install Dependencies

```bash
# create environment
conda create --name fing python==3.10.4
conda activate fing

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install xformers -c xformers/label/dev
pip install transformers accelerate scipy ftfy diffusers[torch]
pip install concurrent-log-handler fastapi fastapi-restful gunicorn uvicorn pydantic 
```

### Get Code and Run

```bash
git clone https://github.com/dkackman/fing.git
cd fing/src
conda activate fing
python -m generator.init_app # only needed once - will take a long time
python -m generator.worker
```

If you see an error about `torch` not being available, leave and re-enter the environment and try again.

```bash
conda deactivate
conda activate fing
```

## Docker

First install the [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on the docker host.

The docker image needs a bind mount for the huggingface model cache. Make sure the host machine has the cached models (typically in `~/.cahce/huggingface`).

```bash
docker build -t dkackman/fing . -f worker.Dockerfile
docker run -it -v "/home/YOUR_USERNAME/.cache/huggingface:/root/.cache/huggingface/" \
    --gpus all \
    --env HUGGINGFACE_TOKEN=YOUR TOKEN \
    --env SDAAS_TOKEN=YOUR TOKEN \
    --env SDAAS_URI=http://fing.kackman.net:9511 \
    dkackman/fing
```
