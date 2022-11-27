# Worker mode

## Install

### Resources

- [Install nvidia drivers ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Ubuntu 22.10

```bash
# install linux nvidia drivers
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall # reboot after
sudo apt install nvtop

# install miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh
conda update conda

# create environment
conda create --name fing python==3.10.4
conda activate fing

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install xformers -c xformers/label/dev
pip install transformers accelerate scipy ftfy diffusers[torch]
pip install concurrent-log-handler fastapi fastapi-restful gunicorn uvicorn pydantic 
```
