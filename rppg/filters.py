import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(signal, fps, low=0.7, high=3.0, order=3):
    if len(signal) < fps * 2:
        return signal

    nyq = 0.5 * fps
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)
