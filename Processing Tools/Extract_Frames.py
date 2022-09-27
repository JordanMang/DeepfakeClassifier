"""
Extracts images from (compressed) videos
Author: Andreas Roessler
Modified by: Jordan Mang
Date: 21.09.2022

This code takes the folders containing deepfake and original videos and extracts still frames from them.
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


DATASET_PATHS = {
    'original': 'original_sequences',
    'Deepfakes': 'manipulated_sequences/Deepfakes'
}
COMPRESSION = ['c0', 'c23', 'c40']


def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == 'cv2':
        count = 0
        vidcap = cv2.VideoCapture(data_path)
        while vidcap.isOpened():
#           adjust the number after count to set the frequency of frame extraction  
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
            success,image = vidcap.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(count)), image)
            count = count + 1
        vidcap.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', '-o', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))
