#!/bin/bash
docker run --gpus all  -v /home/jchan/beepose_fork:/home/beepose/beepose -v /home/jchan/bee:/home/beepose/beepose/bee -it beepose_image 
