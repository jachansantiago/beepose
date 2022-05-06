
import argparse
import cv2
import math
import time
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from beepose.models.train_model import get_testing_model_new
import glob,os

import tensorflow as tf
from beepose.utils.util import NumpyEncoder,save_json,merge_solutions_by_batch
from beepose.inference.inference import inference
import json 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
from keras.models import load_model


def get_model(model_file, gpu=None, gpu_fraction=None):
    
    if gpu is not None:
        if type(gpu)==int:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu

        if gpu_fraction is not None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
            session = tf.Session(config=config)
    
    model = load_model(model_file,compile=False)
    return model

def process_image(image, model, model_config, return_heatmaps=True):
    
    # Adding different threshold for different detection task. For pollen (5,6: 0.09). For tag (0.01)
    params = { 'scale_search':[1], 'thre1': 
              {0:0.4,1:0.25,
               2:0.4,3:0.4,
               4:0.4,5:0.4,
               5:0.09,6:0.09,
               7:0.01}, 
              'thre2': 0.05, 'thre3': 0.4, 
              'min_num': 4, 'mid_num': 10, 
              'crop_ratio': 2.5, 'bbox_ratio': 0.25} 

    model_params = {'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8}
    limbSeq = model_config["skeleton"]
    mapIdx = model_config["mapIdx"]
    np1 = model_config["np1"]
    np2 = model_config["np2"]
    numparts= model_config["numparts"]
    
    print('params:',params,
          'model_params',model_params,
          'np1',np1,
          'np2',np2,
          'resize',4,
          'numparts',numparts,
          'mapIdx',mapIdx,
          'limbSeq',limbSeq) 
    resize_factor=4
    
    show=False
    # generate image with body parts
    #params, model_params = config_reader()
    input_image =cv2.resize(image,(image.shape[1]//resize_factor,image.shape[0]//resize_factor))
    canvas,mappings,parts, heatmaps, paf = inference(input_image, model,params,model_params,show=show,np1=np2,np2=np1,
                                                resize=resize_factor,limbSeq=limbSeq,mapIdx=mapIdx,
                                                numparts=numparts,image_type="BGR", return_heatmaps=True)
    #print(mappings)
    frame_detections={}
    frame_detections['mapping']=mappings
    frame_detections['parts']=parts
    return frame_detections, canvas, heatmaps, paf
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default=None, help='json file name')
    parser.add_argument('--model', type=str, default='models/complete_5p_2.best_day.h5', 
                                help='path to the complete model file model+weights')
    parser.add_argument('--model_config',  default='models/complete_5p_2_model_params.json', type=str, help="Model config json file. This file is created when a training session is started. It should be located in the folder containing the weights.")
#     parser.add_argument('--resize_factor',type=int,default=4)
    
    SIZEMODEL = 4 # Usually I used up to 4.5 GB per model to avoid memory problem when running.
    
    args = parser.parse_args()
    
    image_path = args.image
    output_file = args.output
    if output_file is None:
        filename, _ = os.path.splitext(image_path)
        output_file = filename + ".json"

    filename, _ = os.path.splitext(output_file) 
    output_canvas = filename + ".npy"
    output_heatmaps = filename + "_heatmaps.npy"
    output_paf = filename + "_paf.npy"
    # print(number_models)
    model_file = args.model
    
    config_file = args.model_config
    with open(config_file, 'r') as json_file:
        model_config = json.load(json_file)
        
    print("Loading image..")
    image = cv2.imread(image_path)
    print("Loading model..")
    model = get_model(model_file)
    print("Inference..")
    frame_detections, canvas, heatmaps, paf = process_image(image, model, model_config, return_heatmaps=True)
    
    # print(canvas)
    # print(type(canvas))

    with open(output_canvas, 'wb') as f:
        np.save(f, canvas)

    with open(output_heatmaps, 'wb') as f:
        np.save(f, heatmaps)

    with open(output_paf, 'wb') as f:
        np.save(f, paf)

    with open(output_file, 'w') as outfile:
        json.dump(frame_detections, outfile,cls=NumpyEncoder)
