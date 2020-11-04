import numpy as np
import os

from features.config import VGGISH_FEATURE_DIR

def load_VGGish_features(video_ids, path = VGGISH_FEATURE_DIR):
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        array = np.loadtxt(f"{path}/{filename}")
        if len(array.shape) == 1:
            array = np.reshape(array, (1, array.shape[0]))
        features[int(filename[:-4])].append(array)
    return [np.array(features[video_id])[0] if len(features[video_id]) else np.array([]) for video_id in video_ids]
