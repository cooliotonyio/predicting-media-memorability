import numpy as np

from features.config import C3D_FEATURE_DIR

def load_C3D_features(video_ids, path = C3D_FEATURE_DIR):
    features = []
    for video_id in video_ids:
        filename = f"{path}/{f'{video_id}'.zfill(5)}.mp4.csv"
        features.append(np.loadtxt(filename, delimiter=","))
    return features
