FROM continuumio/anaconda3
WORKDIR /fing
ENV FING_ROOT=/fing/
ENV FING_HOST=0.0.0.0
ENV MODEL_CACHE_DIR=/fing/models

# these should be set with docker run
# ENV HUGGINGFACE_TOKEN
# ENV FING_X_API_KEY # omit to leave x-api-key off

COPY ./src /fing/
RUN conda update -n base -c defaults conda
RUN conda env create -f /fing/environment.yaml

CMD ["conda", "run", "-n", "fing", "python", "-m", "generator.server"]
EXPOSE 9147

#docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> fing