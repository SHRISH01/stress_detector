import numpy as np

class CHROM:
    def __init__(self, fps):
        self.fps = fps
        self.buffer = []
        self.window_size = int(1.6 * fps)

    def update(self, roi_pixels):
        if roi_pixels.size == 0:
            return None

        mean_rgb = roi_pixels.mean(axis=0)
        self.buffer.append(mean_rgb)

        if len(self.buffer) < self.window_size:
            return None

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        rgb = np.array(self.buffer)
        rgb = rgb / np.mean(rgb, axis=0)

        r, g, b = rgb.T

        x = 3 * r - 2 * g
        y = 1.5 * r + g - 1.5 * b
        alpha = np.std(x) / np.std(y)
        signal = x - alpha * y

        return signal
