#!/bin/bash

gunicorn --bind 0.0.0.0:9147 server:gunicorn_app --timeout 120