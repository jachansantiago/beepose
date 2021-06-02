#!/bin/bash
docker run --gpus all -v /home/jchan/beepose/:/home/beepose/output_data -v /mnt/storage/work/jchan/tag_dataset_2019:/home/beepose/input_data -v /mnt/storage/work/jchan/beepose_models:/home/beepose/models -it beepose_image_$(whoami)
# docker run --gpus all -it beepose_image

# docker run --gpus all  -v /home/jchan/beepose_fork:/home/beepose/beepose -v /home/jchan/<input_folder>:/home/beepose/beepose/bee -it beepose_image 
