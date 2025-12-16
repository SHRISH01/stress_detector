import cv2
import numpy as np

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor

from rppg.green import GreenRPPG
from rppg.filters import bandpass


SOURCE = 0
FPS = 30


def main():
    vs = VideoSource(SOURCE)
    vs.open()

    face_detector = FaceLandmarkDetector()
    roi_extractor = ROIExtractor()

    rppg = GreenRPPG(FPS)
    pulse_buffer = []

    while True:
        frame = vs.read()
        if frame is None:
            break

        landmarks, bbox = face_detector.process(frame)

        if landmarks is not None:
            roi_pixels, mask = roi_extractor.extract(frame, landmarks)

            signal = rppg.update(roi_pixels)
            if signal is not None:
                filtered = bandpass(signal, FPS)
                pulse_buffer = filtered[-200:]

            overlay = frame.copy()
            overlay[mask == 255] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if len(pulse_buffer) > 0:
                amp = np.std(pulse_buffer)
                cv2.putText(
                    frame,
                    f"Pulse STD (Green): {amp:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Stress Detector - Green rPPG", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
