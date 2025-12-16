import cv2
import numpy as np

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor

from rppg.green import GreenRPPG
from rppg.filters import bandpass

from physiology.peaks import detect_peaks
from physiology.hrv import compute_hr, rmssd, lf_hf

from stress.index import compute_stress_index


SOURCE = 0
FPS = 30


def main():
    vs = VideoSource(SOURCE)
    vs.open()

    face = FaceLandmarkDetector()
    roi = ROIExtractor()

    rppg = GreenRPPG(FPS)

    pulse_buffer = []
    stress_buffer = []

    last_hr = None
    last_rmssd = None
    last_lfhf = None
    last_stress = None

    while True:
        frame = vs.read()
        if frame is None:
            break

        landmarks, bbox = face.process(frame)

        if landmarks is not None:
            roi_pixels, mask = roi.extract(frame, landmarks)

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
                peaks = detect_peaks(np.array(pulse_buffer), FPS)

                if peaks is not None:
                    rr = np.diff(peaks / FPS)
                    rr = rr[-10:]

                    if len(rr) >= 2:
                        last_hr = compute_hr(rr)
                        last_rmssd = rmssd(rr)

                        lf = lf_hf(rr)
                        if lf is not None:
                            last_lfhf = lf
                            stress = compute_stress_index(last_hr, last_rmssd, lf)
                            stress_buffer.append(stress)
                            stress_buffer = stress_buffer[-20:]
                            last_stress = np.mean(stress_buffer)

        # --------- DISPLAY (ALWAYS) ----------
        y = 30
        if last_hr is not None:
            cv2.putText(frame, f"HR: {last_hr:.1f} BPM", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30

        if last_rmssd is not None:
            cv2.putText(frame, f"RMSSD: {last_rmssd:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30

        if last_lfhf is not None:
            cv2.putText(frame, f"LF/HF: {last_lfhf:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 30

        if last_stress is not None:
            cv2.putText(frame, f"Stress Index: {last_stress:.1f}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Stress Detection - OpenCV", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
