import numpy as np

def normalize(x, xmin, xmax):
    return np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0)

def compute_stress_index(hr, rmssd, hr_history):
    hr_n = normalize(hr, 50, 120)
    rmssd_n = normalize(rmssd, 10, 80)

    if len(hr_history) >= 5:
        hr_slope = hr_history[-1] - hr_history[0]
    else:
        hr_slope = 0.0

    slope_n = normalize(hr_slope, -5, 15)

    stress = (
        0.45 * hr_n +
        0.35 * (1.0 - rmssd_n) +
        0.20 * slope_n
    )

    return 100.0 * stress
