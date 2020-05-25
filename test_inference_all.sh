#!/bin/bash
process_folder_full_video --videos_path bee/videos --GPU 0 1 --GPU_mem=24 --tracking both --model_day models/pose/complete_5p_2.best_day.h5 --model_pollen models/pollen/complete_pollen.h5 --model_config models/pose/complete_5p_2_model_params.json
