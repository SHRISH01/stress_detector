import cv2

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor


def main():
    source = 0
    vs = VideoSource(source)
    vs.open()

    face_detector = FaceLandmarkDetector()
    roi_extractor = ROIExtractor()

    while True:
        frame = vs.read()
        if frame is None:
            break

        landmarks, bbox = face_detector.process(frame)

        if landmarks is not None:
            roi_pixels, mask = roi_extractor.extract(frame, landmarks)

            overlay = frame.copy()
            overlay[mask == 255] = (0, 255, 0)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Stress System - Face & ROI", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
