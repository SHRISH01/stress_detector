import numpy as np

class GreenRPPG:
    def __init__(self, fps):
        self.fps = fps
        self.window = int(1.6 * fps)
        self.buffer = []
        self.window = int(3.0 * fps)

    def update(self, roi_pixels):
        if roi_pixels.size == 0:
            return None

        g_mean = roi_pixels[:, 1].mean()
        self.buffer.append(g_mean)

        if len(self.buffer) < self.window:
            return None

        if len(self.buffer) > self.window:
            self.buffer.pop(0)

        signal = np.array(self.buffer)
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-6)
        return signal