import cv2

class VideoSource:
    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video source")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Video source not opened")
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
