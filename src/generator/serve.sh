#!/bin/bash

gunicorn --bind 0.0.0.0:9147 server:gunicorn_app --timeout 120 --keyfile "/home/don/cert/localhost.key" --certfile "/home/don/cert/localhost.crt"