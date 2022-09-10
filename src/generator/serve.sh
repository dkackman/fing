#!/bin/bash

conda activate fing
gunicorn --bind 0.0.0.0:9147 server:gunicorn_app
