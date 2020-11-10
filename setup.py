import requests
import os
import subprocess
import tqdm
import pandas as pd
import numpy as np
import cv2

# Replace with development_set or testing_set
DATASET = "testing_set"

VIDEO_DIR = f"{DATASET}/Videos"
AUDIO_DIR = f"{DATASET}/Audio"
FRAME_DIR = f"{DATASET}/Frames"

NUM_FRAMES_PER_VIDEO = 8  # number of frame we want to extract

# Replace with dev_video_urls.csv or test_urls.csv
VIDEO_URLS_CSV = f"{DATASET}/test_urls.csv"


def download_video(name, url):
    res = requests.get(url)
    with open(name, "wb") as f:
        for chunk in res.iter_content(chunk_size=255):
            if chunk:
                f.write(chunk)


def extract_audio(video_path, filename):
    subprocess.run(["ffmpeg", "-i", video_path, "-vn", filename],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_total_num_frames(video):
    total = 0
    while True:
        grabbed, _ = video.read()
        if not grabbed:
            break
        total += 1
    return total


def get_frame_indexes(num_frames, n):
    return [int(i) for i in np.around(np.linspace(0, num_frames, n))]


def get_frame_iteratively(video_file, frame_index):
    cap = cv2.VideoCapture(video_file)
    extracted = False
    while not extracted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        extracted, frame = cap.read()
        frame_index -= 1  # try the previous frame

    cap.release()

    return frame, frame_index + 1


def extract_num_frames(video_filename, num_frames, frame_dir=FRAME_DIR, video_dir=VIDEO_DIR):
    video_name = video_filename.split(".")[0].zfill(5)
    video_path = f"{VIDEO_DIR}/{video_filename}"

    cap = cv2.VideoCapture(video_path)

    total_frames = get_total_num_frames(cap)
    frame_indexes = get_frame_indexes(total_frames, num_frames)

    frames_folder = f"{frame_dir}/{video_name}"
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    for frame_index in frame_indexes:
        frame_filename = f"{frames_folder}/{str(frame_index).zfill(3)}.png"

        if not os.path.exists(frame_filename):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            res, frame = cap.read()
            if res == 0:
                frame, frame_index = get_frame_iteratively(
                    video_path, frame_index)
                frame_filename = f"{frames_folder}/{str(frame_index).zfill(3)}.png"
            else:
                cv2.imwrite(frame_filename, frame)

    cap.release()


if __name__ == "__main__":
    if os.path.exists(VIDEO_URLS_CSV):
        # Download videos
        videos = pd.read_csv(VIDEO_URLS_CSV).set_index("video_id")
        if not os.path.exists(VIDEO_DIR):
            os.mkdir(VIDEO_DIR)
        for v, b in tqdm.tqdm(list(videos.iterrows()), desc="Downloading videos"):
            name = f"{VIDEO_DIR}/{v}.mp4"
            if not os.path.exists(name):
                download_video(name, b["video_url"])

        # Extract audio
        if not os.path.exists(AUDIO_DIR):
            os.mkdir(AUDIO_DIR)
        video_filenames = sorted([
            f for f in os.listdir(VIDEO_DIR) if f[-4:] == ".mp4"])
        for video_filename in tqdm.tqdm(video_filenames, desc="Extracting audio"):
            video_path = f"{VIDEO_DIR}/{video_filename}"
            filename = f"{AUDIO_DIR}/{video_filename[:-4]}.wav"
            if not os.path.exists(filename):
                extract_audio(video_path, filename)

        # Extract NUM_FRAMES_PER_VIDEO frames from each video
        if not os.path.exists(FRAME_DIR):
            os.mkdir(FRAME_DIR)
        for filename in tqdm.tqdm(video_filenames, desc=f"Extracting {NUM_FRAMES_PER_VIDEO} frames per video"):
            extract_num_frames(filename, num_frames=NUM_FRAMES_PER_VIDEO)

    else:
        print(f"Could not find {VIDEO_URLS_CSV}!")
    print("Finished setup")
