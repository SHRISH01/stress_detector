import threading
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


class VideoProcessor(threading.Thread):
    def __init__(self, source, state, fps=30):
        super().__init__(daemon=True)
        self.state = state
        self.running = True
        self.fps = fps

        self.vs = VideoSource(source)
        self.face = FaceLandmarkDetector()
        self.roi = ROIExtractor()
        self.rppg = GreenRPPG(fps)

    def run(self):
        self.vs.open()
        pulse_buffer = []
        stress_buffer = []

        while self.running:
            frame = self.vs.read()
            if frame is None:
                break

            landmarks, bbox = self.face.process(frame)
            if landmarks is not None:
                roi_pixels, mask = self.roi.extract(frame, landmarks)

                signal = self.rppg.update(roi_pixels)
                if signal is not None:
                    filtered = bandpass(signal, self.fps)
                    pulse_buffer = filtered[-200:]
                    self.state.pulse.extend(filtered[-5:])

                peaks = detect_peaks(np.array(pulse_buffer), self.fps)
                if peaks is not None:
                    rr = np.diff(peaks / self.fps)
                    if len(rr) >= 2:
                        hr = compute_hr(rr)
                        self.state.hr = hr
                        self.state.hr_hist.append(hr)

                        lf = lf_hf(rr)
                        if lf is not None:
                            stress = compute_stress_index(hr, rmssd(rr), lf)
                            stress_buffer.append(stress)
                            stress_buffer = stress_buffer[-20:]
                            self.state.stress = np.mean(stress_buffer)
                            self.state.stress_hist.append(self.state.stress)

                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.state.frame = frame

        self.vs.release()

    def stop(self):
        self.running = False
