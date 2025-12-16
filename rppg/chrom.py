import numpy as np


class ChromRPPG:
    def __init__(self):
        self.buffer = []

    def update(self, roi_pixels):
        if roi_pixels is None or len(roi_pixels) < 50:
            return None

        r = np.mean(roi_pixels[:, 2])
        g = np.mean(roi_pixels[:, 1])
        b = np.mean(roi_pixels[:, 0])

        self.buffer.append([r, g, b])
        rgb = np.array(self.buffer)

        if len(rgb) < 30:
            return None

        X = 3 * rgb[:, 0] - 2 * rgb[:, 1]
        Y = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]

        X /= np.std(X) + 1e-6
        Y /= np.std(Y) + 1e-6

        return X - Y
