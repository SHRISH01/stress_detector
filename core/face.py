import cv2
import numpy as np
import mediapipe as mp

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.prev_landmarks = None
        self.alpha = 0.7

        self.detector = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        if not result.multi_face_landmarks:
            return None, None

        face_landmarks = result.multi_face_landmarks[0]
        points = []

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        landmarks = np.array(points)

        x1, y1 = landmarks.min(axis=0)
        x2, y2 = landmarks.max(axis=0)

        if self.prev_landmarks is None:
            smooth_landmarks = landmarks
        else:
            smooth_landmarks = (
                self.alpha * self.prev_landmarks +
                (1 - self.alpha) * landmarks
            )

        self.prev_landmarks = smooth_landmarks
        return smooth_landmarks.astype(int), (x1, y1, x2, y2)

        # return landmarks, (x1, y1, x2, y2)
