#!/bin/bash
docker run --gpus all  -v /home/jchan/beepose_fork:/home/beepose/beepose -v /mnt/storage/work/jchan/bees/tag_dataset/:/home/beepose/beepose/bee -it beepose_image
