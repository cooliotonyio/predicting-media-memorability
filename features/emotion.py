import cv2
from fer import FER
import numpy as np
import os
from tqdm import tqdm

from features.config import FRAME_DIR, EMOTION_FEATURE_DIR, EMOTIONS, MAX_FACES


def load_Emotion_features(video_ids, path=EMOTION_FEATURE_DIR):
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        array = np.loadtxt(f"{path}/{filename}")
        if len(array.shape) == 1:
            array = np.reshape(array, (1, array.shape[0]))
        features[int(filename.split(".")[0])] = array
    return [features[video_id] if len(features[video_id]) else np.array([]) for video_id in video_ids]


def build_emotion_vector(emotions_likelihood, max_faces, emotions=EMOTIONS):
    num_faces_frame = len(emotions_likelihood)
    if num_faces_frame > max_faces:  # cap the number of faces
        print(
            f"Video '{video_id}' has {num_faces_frame}, while max_faces is {max_faces}")
        num_faces_frame = max_faces

    frame_feature = []
    for face in range(num_faces_frame):            # detected faces
        frame_feature.append(np.array(
            [emotions_likelihood[face]["emotions"][emotion] for emotion in emotions]))
    for face in range(num_faces_frame, max_faces):  # no faces detected, fill with zero
        frame_feature.append(np.array(np.zeros(len(emotions))))

    emotion_vector = np.concatenate(frame_feature)
    return emotion_vector


def print_extraction_statistics(num_faces_frames, num_faces_videos):
    num_faces_frames = np.array(num_faces_frames)
    num_faces_videos = np.array(num_faces_videos)
    print(f"{np.sum(num_faces_frames > 0)} / {len(num_faces_frames)} frames have faces")
    print(f"{np.sum(num_faces_videos > 0)} / {len(num_faces_videos)} vidoes have faces")
    print("Avg num of faces per frame:", np.mean(num_faces_frames))
    print("Max num of faces per frame:", np.max(num_faces_frames))
    print("Min num of faces per frame:", np.min(num_faces_frames))
    print("Avg num of faces per video:", np.mean(num_faces_videos))
    print("Max num of faces per video:", np.max(num_faces_videos))
    print("Min num of faces per video:", np.min(num_faces_videos))
    print("Avg num of faces per frame for frames with at least 1 face:",
          np.mean(num_faces_frames[num_faces_frames > 1]))
    print("Avg num of faces per video for videos with at least 1 face:",
          np.mean(num_faces_videos[num_faces_videos > 1]))


def extract_emotions(video_ids=None, frame_path=FRAME_DIR, feature_dir=EMOTION_FEATURE_DIR, max_faces=MAX_FACES):
    if video_ids is None:
        video_ids = np.sort(os.listdir(frame_path))

    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    failed = []
    detector = FER()

    num_faces_frames = []
    num_faces_videos = []

    # Extract emotion features
    for video_id in tqdm(video_ids):

        video_features_filename = f"{feature_dir}/{int(video_id)}.csv"
        if os.path.exists(video_features_filename):
            continue

        video_features = []
        num_faces_video = []

        for filename in os.listdir(f"{frame_path}/{video_id}"):
            emotions_likelihood = detector.detect_emotions(
                cv2.imread(f"{frame_path}/{video_id}/{filename}"))

            num_faces_frame = len(emotions_likelihood)
            num_faces_frames.append(num_faces_frame)
            num_faces_video.append(num_faces_frame)

            if num_faces_frame > 0:  # Detected at least one face
                video_features.append(build_emotion_vector(
                    emotions_likelihood, max_faces))
            else:
                failed.append(f"{video_id}/{filename}")

        num_faces_videos.append(np.sum(num_faces_video))

        if video_features:
            video_features = np.array(video_features)
            np.savetxt(video_features_filename, video_features)

    print_extraction_statistics(num_faces_frames, num_faces_videos)

    return failed
