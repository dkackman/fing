# fing
Generative art + chia + nft oh my

## Introduction

This is broken up into two components

### Image generation

This is a python script that runs in a conda environment.

```bash
cd src/generator
conda env create -f environment.yaml
conda activate fing
python generate.py "An impressionist painting of penguin on a bicycle."
```

### REST service

This is a node express webservice. It invokes the generator program and waits for the resulting file to show up in a known location.

```bash
cd src/api
npm install
npm start
```
