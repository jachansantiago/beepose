import argparse
from email.policy import default
import os
from plotbee.video import Video

SKELETON_FOLDER = "/ocean/projects/cis210023p/shared/gurabo10/detections"
VIDEO_FOLDER = "/ocean/projects/cis210023p/shared/gurabo10/videos"

SKELETON_SUFIX = "_skeleton.json"
VIDEO_SUFIX = ".mp4"

OUTPUT_PREFIX = "hungarian_tag_"


def fix_skeleton(video):
    for frame in video:
        for body in frame:
            if (2 in body._parts) and (3 in body._parts):
                body._center_part = 2
                body._angle_conn= [2, 3]
    return video


parser = argparse.ArgumentParser(description='Fix, tagged and track skeleton files from an specific folder.')
parser.add_argument('--colony', type=str, required=True, help='colony to process')
parser.add_argument('--video', type=str, required=True, help='video to process')
parser.add_argument('--workers', type=int, default=5, help="amount of workers")
args = parser.parse_args()

COLONY = args.colony
VIDEO = args.video
WORKERS = args.workers

VIDEO_PATH=f"{VIDEO_FOLDER}/{COLONY}/{VIDEO}{VIDEO_SUFIX}"
SKELETON_PATH=f"{SKELETON_FOLDER}/{COLONY}/{VIDEO}{SKELETON_SUFIX}"
OUTPUT_PATH=f"{SKELETON_FOLDER}/{COLONY}/{OUTPUT_PREFIX}{VIDEO}{SKELETON_SUFIX}"

print(f"Video File: {VIDEO_PATH}")
if not os.path.exists(VIDEO_PATH):
    print("Video not found.")
    exit()

print(f"SKELETON File: {SKELETON_PATH}")
if not os.path.exists(SKELETON_PATH):
    print("Skeleton file not found.")
    exit()

print(f"Loading file {SKELETON_PATH}")
video = Video.load(SKELETON_PATH)
video.load_video(VIDEO_PATH)
    

print(f"Fixing...")
video = fix_skeleton(video)

print(f"Tag detection...")
video.tag_detection(max_workers=WORKERS)
    
video.hungarian_tracking()
    
video.save(OUTPUT_PATH)





