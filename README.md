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

Results!

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
