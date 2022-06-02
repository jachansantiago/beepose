import glob
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create a file with missing video from the specified colony.')

parser.add_argument('--colony', type=str, required=True,
                    help='colony to process')



args = parser.parse_args()
COLONY = args.colony

VIDEO_PATTERN = f"/ocean/projects/cis210023p/shared/gurabo10/videos/{COLONY}/*.mp4"
SKELETON_PATTERN = f"/ocean/projects/cis210023p/shared/gurabo10/detections/{COLONY}/*skeleton*"


def get_video_name(filename):
    _, filename = os.path.split(filename)
    video_name, ext = os.path.splitext(filename)
    return video_name


def get_video_name_from_skeleton_filename(sk_filename):
    _, filename = os.path.split(sk_filename)
    video_name = filename[:-14]
    return video_name


videos_files = glob.glob(VIDEO_PATTERN)
processed_videos = glob.glob(SKELETON_PATTERN)

videos = [get_video_name(v) for v in videos_files]
processed_videos = [get_video_name_from_skeleton_filename(sk) for sk in processed_videos]

missing_videos = list(set(videos) - set(processed_videos))
missing_videos = [v + ".mp4" for v in missing_videos]

np.savetxt(f'/ocean/projects/cis210023p/shared/gurabo10/missing_videos_{COLONY}.txt', np.array(missing_videos), fmt='%s')

print(f"Colony {COLONY} has {len(missing_videos)} video to compute.")