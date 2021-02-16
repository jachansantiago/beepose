#!/bin/bash
process_folder_full_video --videos_path bee/videos --GPU 1 --GPU_mem 12 --tracking hungarian --model_day new_weights/Inference_model.h5 --model_config new_weights/model_params.json
