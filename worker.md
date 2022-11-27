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
```
