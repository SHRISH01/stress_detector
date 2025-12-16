import cv2
from core.face import FaceLandmarkDetector

def test_face_landmarks_detected():
    detector = FaceLandmarkDetector()
    img = cv2.imread("tests/assets/face.jpg")
    landmarks, bbox = detector.process(img)

    assert landmarks is not None
    assert landmarks.shape[0] == 468
    assert bbox is not None
