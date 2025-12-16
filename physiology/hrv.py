import numpy as np
from scipy.signal import welch


def compute_hr(rr_intervals):
    return 60.0 / np.mean(rr_intervals)


def rmssd(rr_intervals):
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff ** 2))


def sdnn(rr_intervals):
    return np.std(rr_intervals)


def lf_hf(rr_intervals, fs=4.0):
    if len(rr_intervals) < 4:
        return None

    rr_interp = np.interp(
        np.linspace(0, len(rr_intervals) - 1, len(rr_intervals) * 4),
        np.arange(len(rr_intervals)),
        rr_intervals
    )

    freqs, psd = welch(rr_interp, fs=fs)

    lf = psd[(freqs >= 0.04) & (freqs < 0.15)].sum()
    hf = psd[(freqs >= 0.15) & (freqs < 0.4)].sum()

    if hf == 0:
        return None

    return lf / hf
