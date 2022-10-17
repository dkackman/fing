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
RUN curl https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -o /tmp/anaconda3.sh
RUN bash /tmp/anaconda3.sh -b -p /anaconda3

RUN /anaconda3/bin/conda init bash
RUN /anaconda3/bin/conda update conda
RUN /anaconda3/bin/conda update -n base -c defaults conda


COPY ./src/environment.yaml /fing/
RUN /anaconda3/bin/conda env create -f /fing/environment.yaml
COPY ./src /fing/
CMD ["/anaconda3/bin/conda", "run", "-n", "fing", "python", "-m", "generator.server"]
EXPOSE 9147

#docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> fing