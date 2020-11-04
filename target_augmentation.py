import pandas as pd
import numpy as np

def add_position_delta(annotations): 
    annotations["t"] = annotations["video_position_second"] - annotations["video_position_first"]

def calculate_alpha(annotations, m_T, T = 75):
    numerator = 0
    denominator = 0
    for video_id, video in annotations.groupby("video_id"):
        t = video["t"].to_numpy()
        x = video["correct"].to_numpy()
        n = len(video)
        m = m_T[video_id]
        for j in range(n):
            numerator += (np.log(t[j] / T) * (x[j] - m)) / n
            denominator += (np.log(t[j] / T) ** 2) / n

    return numerator / denominator

def calculate_memorability(annotations, alpha, T = 75):
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

def calculate_alpha_and_memorability(annotations, T, num_iterations = 10):
    m_T = {video_id: 1 for video_id in annotations["video_id"]}
    for i in range(num_iterations):
        alpha = calculate_alpha(annotations, m_T, T = T)
        m_T = calculate_memorability(annotations, alpha, T = T)
    return alpha, pd.Series(m_T)
