# Predicting Video Memorability (MediaEval 2020)

## Setup

Download training set from google drive and unzip in root

Create conda environment with `conda env create -f environment.yml`

Activate conda enviroment and download videos with `conda activate video-mem && python setup.py`

## Running Jupyter

`conda activate video-mem && jupyter notebook .`

## File structure

- training_set/
  - Features/
  - Videos/
  - video_urls.csv
  - ...
- setup.py
- README.md
- environment.yml