import numpy as np
from scipy.signal import find_peaks


def detect_peaks(signal, fs):
    if len(signal) < fs:
        return None

    distance = int(0.4 * fs)
    peaks, _ = find_peaks(signal, distance=distance)

    if len(peaks) < 2:
        return None

    return peaks
