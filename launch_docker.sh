#!/bin/bash
docker run --gpus all  -v /home/jchan/beepose:/home/beepose/beepose -v /home/jchan/bee:/home/beepose/beepose/bee -it beepose_image

# docker run --gpus all  -v /home/jchan/beepose_fork:/home/beepose/beepose -v /home/jchan/<input_folder>:/home/beepose/beepose/bee -it beepose_image 
