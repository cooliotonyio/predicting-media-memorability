# Predicting Video Memorability (MediaEval 2020)

## Overview

### Abstract

Modeling media memorability has been a consistent challenge in the field of machine learning. The Predicting Media Memorability task in MediaEval2020 is the latest benchmark among similar challenges addressing this topic. Building upon techniques developed in previous iterations of the challenge, we developed ensemble methods with the use of extracted video, image, text, and audio features. Critically, in this work we introduce and demonstrate the efficacy and high generalizability of extracted audio embeddings as a feature for the task of predicting media memorability.

### Links

- [Paper link](https://eigen.no/MediaEval20_paper_53.pdf)
- [Video presentation](https://www.youtube.com/watch?v=6Z_pQe4zm28)
- [Slides](https://docs.google.com/presentation/d/19crIkk-Lg18gDlaSCrlk6WBH4QzmlZjh9jZOa97pc0o/edit#slide=id.p3)

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

Results!

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

```
#############################
SEED: 42
IS_SHORT_TERM: True
spearman_rank            0.345122
test_rank                0.116
p-value                  0.000130
glove_gru                0.450000
vggish_bayesian_ridge    0.350000
c3d_svr                  0.000000
resnet152_svr            0.200000
Name: 0, dtype: float64
SPEARMAN RANK: 0.3451221427293212
#############################
SEED: 1
IS_SHORT_TERM: True
spearman_rank            0.343471
test_rank                0.136
p-value                  0.000140
glove_gru                0.000000
vggish_bayesian_ridge    0.450000
c3d_svr                  0.200000
resnet152_svr            0.350000
Name: 0, dtype: float64
SPEARMAN RANK: 0.3434714172500813
#############################
SEED: 9
IS_SHORT_TERM: True
spearman_rank            0.369693
test_rank                0.085
p-value                  0.000038
glove_gru                0.450000
vggish_bayesian_ridge    0.500000
c3d_svr                  0.050000
resnet152_svr            0.000000
Name: 0, dtype: float64
SPEARMAN RANK: 0.3696931184468572
#############################
SEED: 8
IS_SHORT_TERM: True
spearman_rank            0.357320
test_rank                0.091
p-value                  0.000071
glove_gru                0.350000
vggish_bayesian_ridge    0.150000
c3d_svr                  0.000000
resnet152_svr            0.500000
Name: 0, dtype: float64
SPEARMAN RANK: 0.35731998144759863
#############################
SEED: 7
IS_SHORT_TERM: True
spearman_rank            0.317352
test_rank                0.102
p-value                  0.000462
glove_gru                0.350000
vggish_bayesian_ridge    0.300000
c3d_svr                  0.350000
resnet152_svr            0.000000
Name: 0, dtype: float64
SPEARMAN RANK: 0.31735197338387766
#############################
SEED: 42
IS_SHORT_TERM: False
spearman_rank            0.192480
test_rank                0.076
p-value                  0.036781
glove_gru                0.350000
vggish_bayesian_ridge    0.000000
c3d_svr                  0.550000
resnet152_svr            0.100000
Name: 0, dtype: float64
SPEARMAN RANK: 0.19248036799478244
#############################
SEED: 1
IS_SHORT_TERM: False
spearman_rank            0.288896
test_rank                0.012
p-value                  0.001510
glove_gru                0.450000
vggish_bayesian_ridge    0.150000
c3d_svr                  0.000000
resnet152_svr            0.400000
Name: 0, dtype: float64
SPEARMAN RANK: 0.2888957518744219
#############################
SEED: 9
IS_SHORT_TERM: False
spearman_rank            0.118141
test_rank                0.044
p-value                  0.202613
glove_gru                0.000000
vggish_bayesian_ridge    0.450000
c3d_svr                  0.000000
resnet152_svr            0.550000
Name: 0, dtype: float64
SPEARMAN RANK: 0.11814081090509042
#############################
SEED: 8
IS_SHORT_TERM: False
spearman_rank            0.167716
test_rank                0.077
p-value                  0.069471
glove_gru                0.200000
vggish_bayesian_ridge    0.200000
c3d_svr                  0.250000
resnet152_svr            0.350000
Name: 0, dtype: float64
SPEARMAN RANK: 0.16771562065579423
#############################
SEED: 7
IS_SHORT_TERM: False
spearman_rank            0.201891
test_rank                0.056
p-value                  0.028352
glove_gru                0.650000
vggish_bayesian_ridge    0.000000
c3d_svr                  0.300000
resnet152_svr            0.050000
Name: 0, dtype: float64
SPEARMAN RANK: 0.20189055697738936
```
