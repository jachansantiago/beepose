#!/bin/bash
SHARED=/ocean/projects/cis210023p/shared
IMAGE=$SHARED/simages/beepose.sif
MODEL=$SHARED/models/2019_col2_2stages_model/Inference_model.h5
MODEL_PARAMS=$SHARED/models/2019_col2_2stages_model/model_params.json
WORKERS=5

VIDEO_PATH=$SHARED/gurabo10/videos/col03
OUTPUT=$SHARED/gurabo10/detections/col03

cd $SHARED

# Initialize conda 
source $HOME/.bashrc
conda activate beepose

echo "SIF IMAGE" $IMAGE
echo "MODEL:" $MODEL
echo "MODEL PARAMS:" $MODEL_PARAMS
echo "OUTPUT:" $OUTPUT

singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$1 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 0 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$2 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 1 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$3 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 2 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$4 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 3 --workers $WORKERS &

singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$5 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 4 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$6 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 5 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$7 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 6 --workers $WORKERS &
singularity-beepose --image $IMAGE inference --videos_path $VIDEO_PATH/$8 --model_day $MODEL --model_config $MODEL_PARAMS --output_folder $OUTPUT --GPU 7 --workers $WORKERS &

wait