#!/bin/bash

#SBATCH -N 2
#SBATCH -p GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH -t 2:00:00

#set -x
date

NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))

SHARED=/ocean/projects/cis210023p/shared
IMAGE=$SHARED/simages/beepose.sif
MODEL=$SHARED/models/2019_col2_2stages_model/Inference_model.h5
MODEL_PARAMS=$SHARED/models/2019_col2_2stages_model/model_params.json
OUTPUT=$SHARED/predictions

VIDEO_FOLDER=$SHARED/videos/

WORKERS=5

cd $SHARED

# Initialize conda 
source $HOME/.bashrc
conda activate beepose

echo "SIF IMAGE" $IMAGE
echo "MODEL:" $MODEL
echo "MODEL PARAMS:" $MODEL_PARAMS
echo "OUTPUT:" $OUTPUT

IFS=', ' read -r -a PARSED_VIDEO_LIST<<< "$VIDEO_LIST"
x1=${PARSED_VIDEO_LIST[@]:0:8}
x2=${PARSED_VIDEO_LIST[@]:8:8}

echo "srun -N 1 --ntasks 40 --nodelist=${NODES[0]} ./jobs/pbatch.sh x1"
echo "srun -N 1 --ntasks 40 --nodelist=${NODES[1]} ./jobs/pbatch.sh x2"

wait

