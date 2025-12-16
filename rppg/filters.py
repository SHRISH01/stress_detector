import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    if len(signal) < fs:
        return signal
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)
