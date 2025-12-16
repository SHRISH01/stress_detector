import numpy as np


class GreenRPPG:
    def __init__(self, fps, buffer_size=300):
        self.fps = fps
        self.buffer = []

    def update(self, roi_pixels):
        if roi_pixels is None or len(roi_pixels) < 50:
            return None

        g_mean = np.mean(roi_pixels[:, 1])
        self.buffer.append(g_mean)

        return np.array(self.buffer)
