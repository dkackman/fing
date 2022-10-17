FROM ubuntu:latest
WORKDIR /fing
ENV FING_ROOT=/fing/
ENV FING_HOST=0.0.0.0
ENV MODEL_CACHE_DIR=/fing/models
# these should be set with docker run
# ENV HUGGINGFACE_TOKEN
# ENV FING_X_API_KEY # omit to leave x-api-key off

RUN apt-get update
RUN apt-get -y install curl
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /miniconda

RUN /miniconda/bin/conda init bash
RUN /miniconda/bin/conda update conda
RUN /miniconda/bin/conda update -n base -c defaults conda

COPY ./src/environment.yaml /fing/
RUN /miniconda/bin/conda env create -f /fing/environment.yaml

RUN find /miniconda/ -follow -type f -name '*.a' -delete && \
    find /miniconda/ -follow -type f -name '*.js.map' -delete
RUN /miniconda/bin/conda clean -afy

COPY ./src /fing/
RUN mkdir /fing/models
CMD ["conda", "run", "-n", "fing", "python", "-m", "generator.server"]
EXPOSE 9147

#docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/fing /anaconda3/bin/conda run --live-stream -n fing python -m generator.init_app
#docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> dkackman/fing