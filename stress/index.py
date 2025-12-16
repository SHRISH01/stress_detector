import numpy as np


def normalize(x, xmin, xmax):
    return np.clip((x - xmin) / (xmax - xmin), 0.0, 1.0)


def compute_stress_index(hr, rmssd, lf_hf):
    hr_n = normalize(hr, 50, 120)
    rmssd_n = normalize(rmssd, 10, 80)
    lf_hf_n = normalize(lf_hf, 0.5, 4.0)

    stress = (
        0.35 * hr_n +
        0.35 * lf_hf_n +
        0.30 * (1.0 - rmssd_n)
    )

    return 100.0 * stress
