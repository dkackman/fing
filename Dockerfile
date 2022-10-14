FROM continuumio/anaconda3
WORKDIR /fing
ENV FING_ROOT=/fing/
ENV FING_HOST=0.0.0.0
ENV MODEL_CACHE_DIR=/fing/models

# these should be set with docker run
# ENV HUGGINGFACE_TOKEN
# ENV FING_X_API_KEY # omit to leave x-api-key off

COPY ./src /fing/

#copy or link all of the .pipeline files cached in /tmp for this to work
COPY *.pipeline /fing/models/
RUN conda update -n base -c defaults conda
RUN conda env create -f /fing/environment.yaml
#RUN conda run -n fing python -m generator.init_app
CMD ["conda", "run", "-n", "fing", "python", "-m", "generator.server"]
EXPOSE 9147

#docker run --gpus all --env HUGGINGFACE_TOKEN=<YOUR TOKEN> fing