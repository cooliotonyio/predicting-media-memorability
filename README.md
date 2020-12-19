# Predicting Video Memorability (MediaEval 2020)

## Overview

### Abstract

Modeling media memorability has been a consistent challenge in the field of machine learning. The Predicting Media Memorability task in MediaEval2020 is the latest benchmark among similar challenges addressing this topic. Building upon techniques developed in previous iterations of the challenge, we developed ensemble methods with the use of extracted video, image, text, and audio features. Critically, in this work we introduce and demonstrate the efficacy and high generalizability of extracted audio embeddings as a feature for the task of predicting media memorability.

### Links

- [Paper link](https://eigen.no/MediaEval20_paper_53.pdf)
- [Video presentation](https://www.youtube.com/watch?v=6Z_pQe4zm28)
- [Slides](https://docs.google.com/presentation/d/19crIkk-Lg18gDlaSCrlk6WBH4QzmlZjh9jZOa97pc0o/edit?usp=sharing)

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

## Results

| Team                  | Spearman | Pearson | MSE   |
| --------------------- | -------- | ------- | ----- |
| Memento10k            | 0.137    | 0.13    | 0.01  |
| **UC Berkeley (Us!)** | 0.136    | 0.145   | 0.01  |
| MeMAD                 | 0.101    | 0.09    | 0.01  |
| KT-UPB                | 0.053    | 0.085   | 0.01  |
| Essex-NLIP            | 0.042    | 0.042   | 0.01  |
| DCU@ML-Labs           | 0.034    | 0.078   | 0.10  |
| _Average_             | 0.058    | 0.066   | 0.013 |
| _Variance_            | 0.002    | 0.002   | 0.000 |
