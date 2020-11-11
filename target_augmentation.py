import pandas as pd
import numpy as np


def add_position_delta(annotations):
    t = (annotations["video_position_second"] -
         annotations["video_position_first"]).to_numpy()

    # Some annotations indicate second viewing occured before first
    # which should be impossible, so we'll just flip negative values
    t = np.abs(t)

    # Some annotations indicate first_video and second_video are the same (t=0)
    # which should be impossible, so we'll just assume t=1 for those
    t[t == 0] = 1

    # Some annotations are missing values so that t is NaN, replace with mean
    t = np.nan_to_num(t, nan=np.mean(t[np.logical_not(np.isnan(t))]))

    annotations["t"] = t
    return annotations


def calculate_alpha(annotations, m_T, T=75):
    numerator = 0
    denominator = 0
    for video_id, video in annotations.groupby("video_id"):
        t = video["t"].to_numpy()
        x = video["correct"].to_numpy()
        n = len(video)
        m = m_T[video_id]
        for j in range(n):
            num = (np.log(t[j] / T) * (x[j] - m)) / n
            den = (np.log(t[j] / T) ** 2) / n
            numerator += num
            denominator += den
    return numerator / denominator


def calculate_memorability(annotations, alpha, T=75):
    m_T = {}
    for video_id, video in annotations.groupby("video_id"):
        t = video["t"].to_numpy()
        x = video["correct"].to_numpy()
        n = len(video)
        m = 0
        for j in range(n):
            m += x[j] - (alpha * np.log(t[j] / T))
        m_T[video_id] = m / n
    return m_T


def calculate_alpha_and_memorability(annotations, T, num_iterations=10):
    m_T = {video_id: 1 for video_id in annotations["video_id"].unique()}
    for i in range(num_iterations):
        alpha = calculate_alpha(annotations, m_T, T=T)
        m_T = calculate_memorability(annotations, alpha, T=T)
    return alpha, pd.Series(m_T)
