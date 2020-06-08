import sys 
sys.path.append('..')
import argparse
import os, sys,glob,shutil
import multiprocessing as mp
import cv2
from beepose.utils.merging import merging, merging_pollen
from beepose.tracking.hungarian_tracking import hungarian_tracking, video_hungarian
from beepose.tracking.kalman_tracking import kalman_tracking, video_kalman
from beepose.event_detection.event_detection import do_event_detection_folder,events_update, video_track_classification
from beepose.inference.pollen_detection import pollen_classifier_fragment, pollen_classifier_fragment_skeleton
from beepose.inference.inference_video import process_video_fragment,process_video_by_batch
from beeid.video import Video
import time 
import logging
import json




def process_full_videos(videos_path,model_day,model_nigth,model_pollen,output_folder,sufix,limbSeq,mapIdx,number_models,GPU,GPU_mem,tracking='hungarian',np1=12,np2=6,event_detection=True,process_pollen=True,numparts=5,part='2'):
    
    """
    This Function takes as input  a folder of videos to process. Then takes each of the videos and creates as many subprocesses as possible in order to allocate the maximum possible in the GPU and process the video faster. 
    
    inputs: 
    
        --videos_path : Path where the videos are
        --model: Path where to find the keras trained model. In this version full model + weights should be used. 
        --output_folder: where to save the results
        --sufix: What sufix to use in the results file : NAMEVIDEO_<sufix>.json. For example NAMEVIDEO_detections.json
        --limbSeq: Configuration of the pafs indexing for inference. 
        --mapIdx : Configuration of the skeleton matching configuration for pafs.
        --number_models = How many models to allocate in the GPU
        --GPU : Index of the GPU to use. Usually 0, 1.
        --GPU_mem : Memory available on the gpu to calculate how many models to allocate. 
        --np1 : Number of channels for pafs
        --np2 : Number of channels for parts (heatmaps)
        --tracking: What type of tracking would you like to use: Hungarian, Kalman or both. 
        --process_pollen: Whether to do pollen or not as a separate process. 
    """
    # PARAMETERS FOR TRACKING
    cost_tracking = 200 #HUNGARIAN
    box_size = [400,400] # KALMAN
    profiling={}
    print('==============================================================================')
    print('                              INFERENCE START                                 ')
    print('==============================================================================')
    
    print('Allocating %d Models'%len(GPU)*number_models)
    fraction = 1.0/number_models
    if GPU == 'all':
        fraction = 2.0/number_models
    
    print(fraction)
    np1=np1
    np2=np2
    if output_folder == 'output':
        output_folder = os.path.join(videos_path,'OUTPUT')
        
    os.makedirs(output_folder,exist_ok=True)
    # Creating a folder to move a video once has been processed
    videos_processed = os.path.join(videos_path,'processed')
    os.makedirs(videos_processed,exist_ok=True)
    # creating a folder to put all the processed files
    recicle_folder = os.path.join(output_folder,'recicle')
    os.makedirs(recicle_folder,exist_ok=True)
    
    files_videos = glob.glob(os.path.join(videos_path,'*.mp4'))
    
    detections_files = []
    pollen_files = []
    tracks_files = []
    
    
    
    pool = []
    print('___PROCESSING: ',len(files_videos),' VIDEOS___')
    for ix,file in enumerate(files_videos):
        profiling[file]={}
        print('__processing_video: ',ix,' of ', len(files_videos) )
        # if ix >1: 
        #     try: 
        #         if tracking=='both':
        #             t1.join()
        #             t2.join()
        #         else:
        #             t.join()
        #     except: 
        #         print('Process for tracking not defined or already finished')
        
        tic_total = time.time()
        print('start processing video %s'%file)
        video = cv2.VideoCapture(file)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Video size in frames %d'%num_frames)
        fragment_size = num_frames//(len(GPU)*number_models)
        processes={}
        print('Initializing multiprocessing') 
        filenames = []
        model_num=0
        for G in GPU:
            for i  in range(number_models):
                start = int(model_num*fragment_size)
                end = int(start+fragment_size)
                if model_num == (len(GPU)*number_models)-1:
                    end = int(num_frames)
                output_name = os.path.join(output_folder,'%d'%(model_num+1)+file.split('/')[-1][:-4]+'_%s.json'%sufix)
                filenames.append(output_name)
                processes[model_num] = mp.Process(target=process_video_fragment,args=(file,model_day,G,fraction,start,end,limbSeq,mapIdx,np1,np2,output_name,numparts,int(fragment_size/2)))
                print('starting process%d'%model_num,'File:%s, start:%d, end: %d, gpu:%s, fraction%d'%(file,start,end,str(G),fraction))
                
                processes[model_num].start()
                model_num+=1
           
           
        for k in processes:
            processes[k].join()
        
        print('Video Completed')
        toc_total = time.time()
        print ('Total processing time in this video was %.5f' % (toc_total - tic_total))
        profiling[file]['video']=toc_total - tic_total
        
        
    
        print('______________________')  
        print('Merging all the files')
        print('______________________')
        merged_name = merging(filenames)
        
        
        video_config = {
            'DETECTIONS_PATH' : merged_name,
            'VIDEO_PATH': file
        }
        
        video_skeleton = Video.from_config(video_config)
        video_name = file.split("/")[-1].split('.')[0]
        skeleton_file = video_name + "_skeleton.json"
        skeleton_path = os.path.join(output_folder,  skeleton_file)
        print(skeleton_path)
        video_skeleton.save(skeleton_path)
        print("skeleton file saved.")    
        print(merged_name)

        detections_files.append((skeleton_file, len(video_skeleton)))

    MODEL_SIZE_POLLEN = 2.2 
    if process_pollen:
        print('==============================================================================')
        print('                         Pollen Detection                                      ')
        print('==============================================================================')
        for ix, (skeleton_file, num_frames) in enumerate(detections_files):
            
            model_file_pollen = model_pollen
        
            num_models_pollen_per_gpu = ((GPU_mem/2)//MODEL_SIZE_POLLEN) -1
            fraction = 1/num_models_pollen_per_gpu
            processes={}
            fragment_size = num_frames//(len(GPU)*num_models_pollen_per_gpu)
            pollen_names =[]
            model_num=0
            for G in GPU:
                for i in range(int(num_models_pollen_per_gpu)):
                    start = int(model_num*fragment_size)
                    end = int(start+fragment_size)
                    if model_num == (len(GPU)*num_models_pollen_per_gpu)-1:
                        end = int(num_frames)
                    print("----", start, end)
                    pollen_name = "{:02}".format(model_num) + "pollen_" + skeleton_file
                    pollen_name = os.path.join(output_folder,pollen_name)
                    pollen_names.append(pollen_name)
                    print(pollen_name)
                    tracking_file = os.path.join(output_folder, skeleton_file)
                    processes[model_num] = mp.Process(target=pollen_classifier_fragment_skeleton,args= (tracking_file,pollen_name,model_file_pollen, G,fraction,start,end))
                    processes[model_num].start()
                    model_num +=1
            
            for k in processes:
                processes[k].join()
            
                
            pollen_filename = merging_pollen(pollen_names)
            print(pollen_filename)
            pollen_file = pollen_filename.split("/")[-1]
            pollen_files.append((pollen_file, num_frames))
        
    else:
        pollen_files = detections_files



    print('==============================================================================')
    print('                          %s  Tracking                                        '%tracking)
    print('==============================================================================')
    tic_tracking = time.time()
    tracking_tasks = list()
    if tracking == 'hungarian':
        for ids, (skeleton_file, num_frames) in enumerate(pollen_files):
            tracking_filename = 'hungarian_' + skeleton_file
            tracking_path = os.path.join(output_folder,  tracking_filename)
            tracks_files.append((tracking_filename, num_frames))
            infile = os.path.join(output_folder, skeleton_file)
            t = mp.Process(target=video_hungarian, args=(infile, tracking_path))
            print('Starting hungarian tracking of file %s' % skeleton_file)
            t.start() 
            tracking_tasks.append(t)   
    elif tracking == 'kalman':
        for ids, (skeleton_file, num_frames) in enumerate(pollen_files):
            tracking_filename = 'kalman_' + skeleton_file
            tracking_path = os.path.join(output_folder,  tracking_filename)
            tracks_files.append((tracking_filename, num_frames))
            infile = os.path.join(output_folder, skeleton_file)
            t = mp.Process(target=video_kalman, args=(infile, tracking_path))
            print('Starting kalman tracking of file %s' % skeleton_file)
            t.start()
            tracking_tasks.append(t)  
    elif tracking == 'both':
        for ids, (skeleton_file, num_frames) in enumerate(pollen_files):
            tracking_filename = 'hungarian_' + skeleton_file
            tracking_path = os.path.join(output_folder,  tracking_filename)
            tracks_files.append((tracking_filename, num_frames))
            infile = os.path.join(output_folder, skeleton_file)
            t1 = mp.Process(target=video_hungarian, args=(infile, tracking_path))
            print('Starting hungarian tracking of file %s' % skeleton_file)
            t1.start()
            tracking_tasks.append(t1) 
        
            tracking_filename = 'kalman_' + skeleton_file
            tracking_path = os.path.join(output_folder,  tracking_filename)
            tracks_files.append((tracking_filename, num_frames))
            infile = os.path.join(output_folder, skeleton_file)
            t2 = mp.Process(target=video_kalman, args=(infile, tracking_path))
            print('Starting kalman tracking of file %s' % skeleton_file)
            t2.start()
            tracking_tasks.append(t2) 
    # try: 
    #     if tracking=='both':
    #         t1.join()
    #         t2.join()
    #     else:
    #         t.join()
    # except: 
    #     print('Process for tracking not defined or already finished') 
    for task in tracking_tasks:
        task.join()
    #toc_tracking = time.time()    
    #profiling[file]['tracking']=toc_tracking-tic_tracking
    print('==============================================================================')
    print('                         Event Detection                                      ')
    print('==============================================================================')
    if event_detection: 
        event_files = list()
        procs = list()
        for ix,(track_file,num_frames) in enumerate(tracks_files):
            outfile = 'event_' + track_file
            event_name = os.path.join(output_folder, outfile)

            infile = os.path.join(output_folder, track_file)
            print('Event detection on file %s'% event_name)
            t = mp.Process(target=video_track_classification,args=(infile, event_name))
            t.start()
            procs.append(t)
            event_files.append((outfile, num_frames))

        for p in procs:
            p.join()
            
       
    print('==============================================================================')
    print('                              INFERENCE DONE                                  ')
    print('==============================================================================')
        
    
        
    # print('Cleaning space and moving video to processed folder and chunks of detections to recycle')
    # print('Profiling:',profiling)
    # for f in filenames:
    #     shutil.move(f,recicle_folder)

        



def process_full_videos_by_batch(videos_path,model_day,model_nigth,model_pollen,output_folder,sufix,limbSeq,mapIdx,number_models,GPU,GPU_mem,tracking='hungarian',np1=12,np2=6):
    
    """
    This Function takes as input  a folder of videos to process. Then it uses a safer model than process full videos, using dasks to do the parallel experimentation. 
    
    inputs: 
    
        --videos_path : Path where the videos are
        --model: Path where to find the keras trained model. In this version full model + weights should be used. 
        --output_folder: where to save the results
        --sufix: What sufix to use in the results file : NAMEVIDEO_<sufix>.json. For example NAMEVIDEO_detections.json
        --limbSeq: Configuration of the pafs indexing for inference. 
        --mapIdx : Configuration of the skeleton matching configuration for pafs.
        --number_models = How many models to allocate in the GPU
        --GPU : Index of the GPU to use. Usually 0, 1.
        --GPU_mem : Memory available on the gpu to calculate how many models to allocate. 
        --np1 : Number of channels for pafs
        --np2 : Number of channels for parts (heatmaps)
        --tracking: What type of tracking would you like to use: Hungarian, Kalman or both. 
    """
    # PARAMETERS FOR TRACKING
    cost_tracking = 200 #HUNGARIAN
    box_size = [400,400] # KALMAN
    resize_factor =4
    
    print('==============================================================================')
    print('                              INFERENCE START                                 ')
    print('==============================================================================')
    
    print('Allocating %d Models'%number_models)
    fraction = 1.0/number_models
    if GPU =='all':
        fraction=2.0/number_models
    
    print(fraction)
    np1=np1
    np2=np2
    if output_folder == 'output':
        output_folder = os.path.join(videos_path,'OUTPUT')
        
    os.makedirs(output_folder,exist_ok=True)
    # Creating a folder to move a video once has been processed
    videos_processed = os.path.join(videos_path,'processed')
    os.makedirs(videos_processed,exist_ok=True)
    # creating a folder to put all the processed files
    recicle_folder = os.path.join(output_folder,'recicle')
    os.makedirs(recicle_folder,exist_ok=True)
    
    files_videos = glob.glob(os.path.join(videos_path,'*.mp4'))
    
    
    
    pool = []
    print('___PROCESSING: ',len(files_videos),' VIDEOS___')
    for ix,file in enumerate(files_videos):
        print('__processing_video: ',ix,' of ', len(files_videos) )
        if ix >1: 
            try: 
                if tracking=='both':
                    t1.join()
                    t2.join()
                else:
                    t.join()
            except: 
                print('Process for tracking not defined or already finished')
        
        tic_total = time.time()
        print('start processing video %s'%file)
        video = cv2.VideoCapture(file)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        chunk=150
        output_name = os.path.join(output_folder,'merged_'+file.split('/')[-1][:-4]+'_%s.json'%sufix)
        init_frame=0
        process_video_by_batch(video,model_day,model_nigth,GPU,init_frame,num_frames,chunk,output_name,limbSeq,mapIdx,resize_factor=resize_factor,np1=np1,np2=np2)
       
        merged_name=output_name
        print('Video Completed')
        toc_total = time.time()
        print ('Total processing time in this video was %.5f' % (toc_total - tic_total))

        print('==============================================================================')
        print('                          %s  Tracking                                        '%tracking)
        print('==============================================================================')
        
        if tracking == 'hungarian':
        
            t = mp.Process(target=hungarian_tracking,args=(merged_name,cost_tracking))
            print('Starting tracking of file %s'%merged_name)
            t.start()
            
        elif tracking == 'kalman':
            t = mp.Process(target=kalman_tracking,args=(merged_name,'',box_size))
            print('Starting tracking of file %s'%merged_name)
            t.start()
        elif tracking == 'both':
            t1 = mp.Process(target=kalman_tracking,args=(merged_name,'',box_size))
            print('Starting tracking of file %s'%merged_name)
            t1.start()
            t2 = mp.Process(target=hungarian_tracking,args=(merged_name,cost_tracking))
            print('Starting tracking of file %s'%merged_name)
            t2.start()
    try: 
        if tracking=='both':
            t1.join()
            t2.join()
        else:
            e.join()
    except: 
        print('Process for tracking not defined or already finished') 
        
        
    print('==============================================================================')
    print('                         Event Detection                                      ')
    print('==============================================================================')
        
    if tracking == 'hungarian':
         
        prefix = 'id_nms_track_'
        print('Event detection on file %s'%prefix)
        e = mp.Process(target=do_event_detection_folder,args=(prefix,'',output_folder,72000))
        e.start()
        e.join()
    elif tracking == 'kalman':
        prefix = 'id_kalman_tracks.json' 
        print('Event detection on file %s'%prefix)
        e = mp.Process(target=do_event_detection_folder,args=(prefix,'',output_folder,72000))
        e.start()
        e.join()
    elif tracking == 'both':
        prefix = 'id_nms_track_'
        print('Event detection on file %s'%prefix)
        e1 = mp.Process(target=do_event_detection_folder,args=(prefix,'',output_folder,72000))
        e1.start()
        prefix = 'id_kalman_tracks.json' 
        print('Event detection on file %s'%prefix)
        e2 = mp.Process(target=do_event_detection_folder,args=(prefix,'',output_folder,72000))
        e2.start()
        e1.join()
        e2.join()
            
    print('==============================================================================')
    print('                         Pollen Detection                                      ')
    print('==============================================================================')
    
    MODEL_SIZE_POLLEN = 2.2 
    for ix,file in enumerate(files_videos):
        
        
        det_name= 'merged_'+file.split('/')[-1][:-4]+'_detections.json'
        det_file = os.path.join(output_folder,det_name)
        if tracking in ['hungarian','both']:
            trk_file = os.path.join(output_folder,'track_nms_'+det_name)
            id_trk_file = os.path.join(output_folder,'id_nms_track_'+det_name)
        else:
            trk_file = os.path.join(output_folder,det_name[:-4]+'kalman_tracks.json')
            id_trk_file = os.path.join(output_folder,det_name[:-4]+'id_kalman_tracks.json')
            
        model_file_pollen = model_pollen
        
        num_models_pollen = GPU_mem//MODEL_SIZE_POLLEN
        
        fraction = 1/num_models_pollen
        processes={}
        video = cv2.VideoCapture(file)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fragment_size = num_frames//num_models_pollen
        for i in range(int(num_models_pollen)):
            start = int(i*fragment_size)
            end = int(start+fragment_size)
            if i == number_models-1:
                end = int(num_frames)
            processes[i] = mp.Process(target=pollen_classifier_fragment,args= (det_file,trk_file,file, model_file_pollen, GPU,fraction,start,end))
            processes[i].start()
            
        for k in processes:
            processes[k].join()
        
        pollen_file = os.path.join(output_folder,'trk_pollen_'+det_name)
        count_file = os.path.join(output_folder,'Count_v2_id_nms_track_'+det_name)
        trk_class_file = os.path.join(output_folder,'TRK_Class_id_nms_track_'+det_name)
        
        events_update(pollen_file,id_trk_file,count_file,trk_class_file)
        
        print('Video Ready to be moved')
        shutil.move(file,videos_processed)
       
    print('==============================================================================')
    print('                              INFERENCE DONE                                  ')
    print('==============================================================================')
        
    
        
    print('Cleaning space and moving video to processed folder and chunks of detections to recycle')
    
    #for f in filenames:
    #    shutil.move(f,recicle_folder)
    
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', type=str, help='input folder path where the videos are')
    parser.add_argument('--GPU',default=[0], nargs='+', help="GPU number for the device. If you want to use more than one, separate by commas like 0,1,2 etc")
    parser.add_argument('--GPU_mem',type = float, default=12, help='Memory available')
    parser.add_argument('--model_day', default='../../models/pose/complete_5p_2.best_day.h5', type=str, help='path to day model')
    parser.add_argument('--model_nigth', default='../../models/pose/complete_5p_2.best_night.h5', type=str, help='path to night model')
    parser.add_argument('--model_pollen', default='../../models/pollen/complete_pollen.h5', type=str, help='path to night model')
    parser.add_argument('--output_folder', default='output',type=str)
    # TODO_FIX THIS PARSER. FOR NOW JUST HARDCODED. 
    #parser.add_argument('--limb_conf',type=list,default=[[1,3],[3,2],[2,4],[2,5],[1,2]] )
    #parser.add_argument('--paf_conf',type=list,default=[[0,1],[2,3],[4,5],[6,7],[8,9]] )
    parser.add_argument('--sufix', type=str, default= 'detections', help='Sufix to identify the detection ')
    parser.add_argument('--tracking',type=str, default= 'hungarian', choices= ['hungarian','kalman','both'])
#     parser.add_argument('--np1',type=int,default=12, help = 'number of channels for pafs')
#     parser.add_argument('--np2',type=int,default=6, help= 'number of channels for heatmaps')
#     parser.add_argument('--numparts',type=int,default=5, help='number of parts to process')
    parser.add_argument('--model_config', default='../../models/pose/complete_5p_2_model_params.json', type=str, help="Model config json file")
    parser.add_argument('--part',type=int,default=2, help='Index id of Part to be tracked')
    parser.add_argument('--process_pollen', default=True, action="store_true", help='Whether to apply pollen detection separately. Default is True')
    parser.add_argument('--event_detection', default=True, action="store_true", help='Whether to apply event detection. Default is True')
    parser.add_argument('--debug',type=bool,default=False,help='If debug is True logging will include profiling and other details')
    SIZEMODEL = 4 # Usually I used up to 4.5 GB per model to avoid memory problem when running.
    
    args = parser.parse_args()
     
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
      

    
    print(args)
    try:
        GPU = [int(g) for g in args.GPU]
    except:
        GPU = args.GPU
        
    GPU_mem = args.GPU_mem
    if GPU=='all':
        GPU_mem +=GPU_mem
    
    videos_path = args.videos_path
    output_folder = args.output_folder
    number_models_per_gpu = int(GPU_mem//SIZEMODEL)
    # print(number_models)
    model_day = args.model_day
    model_nigth = args.model_nigth
    model_pollen = args.model_pollen
    
    
    config_file = args.model_config
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    print(config)
    
    limbSeq = config["skeleton"]
    mapIdx = config["mapIdx"]
    sufix = args.sufix
    tracking = args.tracking
    np1 = config["np1"]
    np2 = config["np2"]
    numparts= config["numparts"]
    part = str(args.part)
    process_pollen = args.process_pollen
    event_detection = args.event_detection
    
    # Slower than I thought. It may be improved. But need to parellize
    #process_full_videos_by_batch(videos_path,model_day,model_nigth,model_pollen,output_folder,sufix,limbSeq,mapIdx,number_models,GPU,GPU_mem,tracking=tracking,np1=np1,np2=np2)
    process_full_videos(videos_path,model_day,model_nigth,model_pollen,output_folder,sufix,limbSeq,mapIdx,number_models_per_gpu,GPU,GPU_mem,tracking=tracking,np1=np1,np2=np2,event_detection=event_detection,process_pollen=process_pollen,numparts=numparts)
    
if __name__ == '__main__':
    main()
