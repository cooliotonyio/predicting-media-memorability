import requests
import os
import tqdm
import pandas as pd


def download_video(name, url):
    res = requests.get(url)
    with open(name, "wb") as f:
        for chunk in res.iter_content(chunk_size=255):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    if os.path.exists("training_set/video_urls.csv"):
        videos = pd.read_csv(
            "training_set/video_urls.csv").set_index("video_id")
        if not os.path.exists("training_set/Videos"):
            os.mkdir("training_set/Videos")
        for v, b in tqdm.tqdm(list(videos.iterrows()), desc="Downloading videos"):
            name = f"training_set/Videos/{v}.mp4"
            if not os.path.exists(name):
                download_video(name, b["video_url"])

    else:
        print("Could not find training_set!")
    print("Finished downloading videos")
