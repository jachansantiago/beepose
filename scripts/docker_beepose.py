import argparse
import os
import subprocess
from pathlib import Path
import docker
from docker.types import Mount


#GLOBAL VARIABLES
USER = os.environ["USER"]
BEEPOSE_IMAGE = "beepose_image_{user}".format(user=USER)
# print(BEEPOSE_IMAGE)

TRAINING_DATA_OPTIONS = ["ann", "imgs", "val_imgs", "val_ann"]
TRAINING_OUTPUT_OPTIONS = ["folder"]

INFERENCE_DATA_OPTIONS = ["videos_path"]
INFERENCE_MODEL_OPTIONS = ["model_day", "model_config"]
INFERENCE_OUTPUT_OPTIONS = ["output_folder"]

device_requests=[
        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
    ]

def force_dir(folder):
    if os.path.isdir(folder) and folder[-1] != "/":
        folder += "/"
    return folder

def use_dir(folder):
    if not os.path.isdir(folder):
        folder, file = os.path.split(folder)
    return folder

def absolute_path(path):
    os.makedirs(path, exist_ok=True)
    return Path(path).resolve()


def execute_job(cmd, mounts):
    client = docker.from_env()
    container = client.containers.run(BEEPOSE_IMAGE, cmd, 
                                      mounts=mounts, 
                                      device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                                      detach=True)
    for line in container.logs(stream=True):
        print(line.strip().decode("utf-8"))


def beepose_args_options(beepose_args):
    options = list()
    for key, value in beepose_args.items():
        if key == "command":
            continue
        options.append("--{}".format(key))
        if key == "GPU":
            options.append(" ".join(value))
        else:
            options.append("{}".format(value))
    return options

def get_docker_path(file_path, docker_path):
    path, filename = os.path.split(file_path)
    return os.path.join(docker_path, filename)

def get_mount(file_path, docker_path):
    return Mount(str(docker_path), str(file_path), type="bind")


def get_mounts(args_dict, options, docker_folder):
    mounts = list()
    
    for option in options:
        host_path = args_dict[option]
        docker_path = get_docker_path(host_path, docker_folder)

        mount = get_mount(host_path, docker_path)
        mounts.append(mount)
    return mounts

def update_beepose_args(args_dict, options, docker_folder):
    volumens = list()

    beepose_args = args_dict.copy()
    
    for option in options:
        host_path = args_dict[option]
        docker_path = get_docker_path(host_path, docker_folder)
        beepose_args[option] = docker_path

    return beepose_args


def launch_training(args):
    args_dict = vars(args)
    training_data_mounts = get_mounts(args_dict, TRAINING_DATA_OPTIONS, "/training_data")
    training_output_mounts = get_mounts(args_dict, TRAINING_OUTPUT_OPTIONS, "/training_model")
    
    mounts = training_data_mounts + training_output_mounts

    # update beepose args path inside the docker container
    beepose_args = update_beepose_args(args_dict, TRAINING_DATA_OPTIONS, "/training_data")
    beepose_args = update_beepose_args(beepose_args, TRAINING_OUTPUT_OPTIONS, "/training_model")

    cmd = [ "train"]
    cmd += beepose_args_options(beepose_args)
    cmd = " ".join(cmd)
    
    execute_job(cmd, mounts)

def launch_inference(args):
    args_dict = vars(args)
    args_dict["videos_path"] = force_dir(args_dict["videos_path"])
    args_dict["output_folder"] = absolute_path(args_dict["output_folder"])
    inference_data_mounts = get_mounts(args_dict, INFERENCE_DATA_OPTIONS, "/inference_data")
    inference_model_mounts = get_mounts(args_dict, INFERENCE_MODEL_OPTIONS , "/inference_model")
    inference_output_mounts = get_mounts(args_dict, INFERENCE_OUTPUT_OPTIONS , "/inference_output")
    
    mounts = inference_data_mounts + inference_model_mounts + inference_output_mounts

    beepose_args = update_beepose_args(args_dict, INFERENCE_DATA_OPTIONS, "/inference_data")
    beepose_args = update_beepose_args(beepose_args, INFERENCE_MODEL_OPTIONS , "/inference_model")
    beepose_args = update_beepose_args(beepose_args, INFERENCE_OUTPUT_OPTIONS , "/inference_output")
    beepose_args["videos_path"], _ = os.path.split(beepose_args["videos_path"])

    cmd = ["inference"]
    cmd += beepose_args_options(beepose_args)
    cmd = " ".join(cmd)
    
    execute_job(cmd, mounts)


# create the top-level parser
parser = argparse.ArgumentParser()
parser.set_defaults(command=None)
subparsers = parser.add_subparsers()

# create the parser for the "skeleton" command
training_parser = subparsers.add_parser('train')
training_parser.add_argument('--stages', type=int, default =2, help='number of stages')
training_parser.add_argument('--folder',type=str,default="weights_logs/5p_6/",help='"Where to save this training"' )
training_parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
training_parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
training_parser.add_argument('--ann', type=str,default = '../../data/raw/bee/dataset_raw/train_bee_annotations2018.json' ,help =' Path to annotations')
training_parser.add_argument('--imgs',type=str, default = '../../data/raw/bee/dataset_raw/train',help='Path to images folder')
training_parser.add_argument('--val_imgs',type=str,default='../../data/raw/bee/dataset_raw/validation',help= 'path to val images folder')
training_parser.add_argument('--val_ann',type=str,default='../../data/raw/bee/dataset_raw/validation.json',help= 'path to val images folder')
training_parser.add_argument('--batch_size', type=int, default =10, help= 'batch_size' )
training_parser.add_argument('--max_iter', type=int,default=20000, help='Number of epochs to run ')
training_parser.set_defaults(command="train")

# create the parser for the "inference" command
inference_parser = subparsers.add_parser('inference')
inference_parser.add_argument('--videos_path', type=str, help='input folder path where the videos are')
inference_parser.add_argument('--GPU',default=[0], nargs='+', help="GPU number for the device. If you want to use more than one, separate by commas like 0,1,2 etc")
inference_parser.add_argument('--GPU_mem',type = float, default=12, help='Memory available')
inference_parser.add_argument('--model_day', default='../../models/pose/complete_5p_2.best_day.h5', type=str, help='path to day model')
inference_parser.add_argument('--output_folder', default="output",type=str)
inference_parser.add_argument('--model_config', default='../../models/pose/complete_5p_2_model_params.json', type=str, help="Model config json file")
inference_parser.set_defaults(command="inference")


#parse the args and call whatever function was selected
args = parser.parse_args()
if args.command == None:
    parser.parse_args(['-h'])
elif args.command == "train":
    launch_training(args)
elif args.command == "inference":
    launch_inference(args)





