import requests
import os
import subprocess
import tqdm
import pandas as pd

VIDEO_DIR = "training_set/Videos"
AUDIO_DIR = "training_set/Audio"


def download_video(name, url):
    res = requests.get(url)
    with open(name, "wb") as f:
        for chunk in res.iter_content(chunk_size=255):
            if chunk:
                f.write(chunk)


def extract_audio(video_path, filename):
    subprocess.run(["ffmpeg", "-i", video_path, "-vn", filename],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    if os.path.exists("training_set/video_urls.csv"):

        # Download videos
        videos = pd.read_csv(
            "training_set/video_urls.csv").set_index("video_id")
        if not os.path.exists(VIDEO_DIR):
            os.mkdir(VIDEO_DIR)
        for v, b in tqdm.tqdm(list(videos.iterrows()), desc="Downloading videos"):
            name = f"{VIDEO_DIR}/{v}.mp4"
            if not os.path.exists(name):
                download_video(name, b["video_url"])

        # Extract audio
        if not os.path.exists(AUDIO_DIR):
            os.mkdir(AUDIO_DIR)
        video_filenames = [
            f for f in os.listdir(VIDEO_DIR) if f[-4:] == ".mp4"]
        for video_filename in tqdm.tqdm(video_filenames, desc="Extracting audio"):
            video_path = f"{VIDEO_DIR}/{video_filename}"
            filename = f"{AUDIO_DIR}/{video_filename[:-4]}.wav"
            if not os.path.exists(filename):
                extract_audio(video_path, filename)

    else:
        print("Could not find training_set!")
    print("Finished setup")
