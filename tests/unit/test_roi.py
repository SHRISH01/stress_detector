import cv2
import numpy as np
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor

def test_roi_extraction():
    img = cv2.imread("tests/assets/face.jpg")
    detector = FaceLandmarkDetector()
    roi_extractor = ROIExtractor()

    landmarks, _ = detector.process(img)
    roi_pixels, mask = roi_extractor.extract(img, landmarks)

    assert roi_pixels.shape[0] > 1000
    assert mask.sum() > 0
