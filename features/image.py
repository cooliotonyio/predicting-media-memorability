import numpy as np
import os

from features.config import RESNET152_FEATURE_DIR, HOG_FEATURE_DIR, LBP_FEATURE_DIR

def load_ResNet152_features(video_ids, path = RESNET152_FEATURE_DIR):
    features = {video_id : [] for video_id in video_ids}
    for filename in os.listdir(path):
        features[int(filename[:5])].append(np.loadtxt(f"{path}/{filename}"))
    return [np.array(features[video_id])[0] for video_id in video_ids]

def load_HOG_features(video_ids, path = HOG_FEATURE_DIR):
    '''Histogram of Oriented Gradients'''
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        video_id = int(filename[:5])
        features[video_id].append(np.loadtxt(f"{path}/{filename}", delimiter=","))
    return [np.array(features[video_id]) for video_id in video_ids]

def load_LBP_features(video_ids, path = LBP_FEATURE_DIR):
    '''Local Binary Pattern'''
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        video_id = int(filename[:5])
        features[video_id].append(np.loadtxt(f"{path}/{filename}", delimiter=","))
    return [np.array(features[video_id]) for video_id in video_ids]
