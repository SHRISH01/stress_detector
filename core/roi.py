import numpy as np
import cv2

FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356]
LEFT_CHEEK_IDX = [50, 187, 207, 216, 192]
RIGHT_CHEEK_IDX = [280, 411, 427, 436, 416]

class ROIExtractor:
    def __init__(self):
        self.regions = {
            "forehead": FOREHEAD_IDX,
            "left_cheek": LEFT_CHEEK_IDX,
            "right_cheek": RIGHT_CHEEK_IDX
        }

    def extract(self, frame, landmarks):
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for idxs in self.regions.values():
            pts = landmarks[idxs]
            cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)

        roi_pixels = frame[mask == 255]
        return roi_pixels, mask
