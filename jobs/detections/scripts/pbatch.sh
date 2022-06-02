#!/bin/bash


SHARED=/ocean/projects/cis210023p/shared
IMAGE=$SHARED/simages/beepose.sif
MODEL=$SHARED/models/2019_col2_2stages_model/Inference_model.h5
MODEL_PARAMS=$SHARED/models/2019_col2_2stages_model/model_params.json
WORKERS=5

VIDEO_FOLDER=$SHARED/gurabo10/videos/col01/
OUTPUT=$SHARED/gurabo10/detections/col01/



singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$1 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 0 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$2 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 1 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$3 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 2 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$4 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 3 --workers $WORKERS &

singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$5 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 4 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$6 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 5 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$7 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 6 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_FOLDER$8 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 7 --workers $WORKERS &
