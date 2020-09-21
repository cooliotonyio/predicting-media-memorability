# Predicting Video Memorability (MediaEval 2020)

## Setup

Download training set from google drive and unzip in root

Create conda environnment with `conda env create -f environment.yml`

### Download videos and extract audio

Activate conda environment with `conda activate video-mem`
Run setup with `python setup.py` (downloads videos and extracts audio)
Audio extraction requires command line access to [FFmpeg](https://ffmpeg.org/)

## Running a Jupyter Server

`conda activate video-mem && jupyter notebook .`

## File structure

```bash
- training_set/         #Data belongs here
  - Features/
  - Videos/             #Created by setup.py
  - Audio/              #Created by setup.py
  - video_urls.csv
  - ...
- setup.py              #Downloads videos
- README.md
- environment.yml
- train.ipynb           #Training workflow
```
