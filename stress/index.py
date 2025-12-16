import numpy as np

def normalize(x, xmin, xmax):
    return np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0)

def compute_stress_index(hr, rmssd, hr_history):
    # --- Relative HR ---
    if len(hr_history) >= 10:
        baseline_hr = np.mean(hr_history[:-1])
        hr_delta = hr - baseline_hr
        hr_n = normalize(hr_delta, -12, 18)
    else:
        hr_n = 0.4

    # --- RMSSD (primary stress suppressor) ---
    rmssd_n = normalize(rmssd, 15, 80)

    # --- Smoothed HR slope (reduces laughter noise) ---
    if len(hr_history) >= 5:
        slope = np.mean(np.diff(hr_history[-5:]))
    else:
        slope = 0.0
    slope_n = normalize(slope, -3, 6)

    # --- Parasympathetic override ---
    parasymp_factor = 1.0 - rmssd_n

    stress = (
        0.25 * hr_n +
        0.35 * parasymp_factor +
        0.15 * slope_n
    )

    # --- Strong suppression if RMSSD is high ---
    if rmssd_n > 0.65:
        stress *= 0.6

    return np.clip(100.0 * stress, 0, 100)
