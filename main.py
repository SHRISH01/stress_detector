import cv2
import numpy as np

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor
from rppg.green import GreenRPPG
from rppg.filters import bandpass

FPS = 30


def main():
    vs = VideoSource(0)
    vs.open()

    face = FaceLandmarkDetector()
    roi = ROIExtractor()
    rppg = GreenRPPG(FPS)

    while True:
        frame = vs.read()
        if frame is None:
            break

        landmarks, bbox = face.process(frame)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if landmarks is not None:
            roi_pixels, mask = roi.extract(frame, landmarks)

            overlay = frame.copy()
            overlay[mask == 255] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            signal = rppg.update(roi_pixels)
            if signal is not None and len(signal) > FPS:
                filtered = bandpass(signal, FPS)
                bpm = np.argmax(np.abs(np.fft.rfft(filtered))) * FPS / len(filtered)
                cv2.putText(frame, f"HR peak ~ {bpm:.1f} Hz",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

        cv2.imshow("ROI + Face Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
