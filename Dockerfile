FROM continuumio/anaconda3
WORKDIR /fing
COPY ./src /fing/
RUN conda init bash
RUN conda update -n base -c defaults conda
RUN conda env create -f /fing/environment.yaml
RUN source $(conda info --base)/etc/profile.d/conda.sh
RUN conda activate fing
RUN python -m /fing/generator.init_app
CMD ["python", "-m", "/fing/generator.server"]
EXPOSE 9147
