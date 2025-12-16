import sys
import cv2
import numpy as np
import time
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QFrame
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor
from rppg.green import GreenRPPG
from rppg.filters import bandpass
from physiology.peaks import detect_peaks
from physiology.hrv import compute_hr, rmssd
from stress.index import compute_stress_index


FPS = 30
GRAPH_UPDATE_INTERVAL = 1000 
SUMMARY_INTERVAL = 30  


PLOTLY_HTML = """
<html>
<head>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body style="margin:0;background:#111;">
<div id="plot"></div>
<script>
var layout = {
    title: "%s",
    height: 180,             // 🔹 CHANGE THIS
    paper_bgcolor: "#111",
    plot_bgcolor: "#111",
    font: {color: "#ddd"},
    margin: {l:35,r:15,t:35,b:25}
};
var trace = { y: [], mode: "lines", line: {color: "%s"} };
Plotly.newPlot("plot", [trace], layout);
</script>
</body>
</html>
"""

class StressDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Stress Monitoring Dashboard")
        self.setGeometry(50, 50, 1500, 850)

        # --------- VIDEO PIPELINE --------- #
        self.vs = None
        self.face = FaceLandmarkDetector()
        self.roi = ROIExtractor()
        self.rppg = GreenRPPG(FPS)

        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)

        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.update_graphs)

        # --------- BUFFERS --------- #
        self.pulse_buffer = []
        self.hr_history = deque(maxlen=60)
        self.rmssd_history = deque(maxlen=60)
        self.stress_history = deque(maxlen=60)

        self.last_hr = None
        self.last_rmssd = None
        self.last_stress = None

        self.start_time = time.time()

        self.init_ui()
        self.init_graphs()
        self.apply_theme()

    # ---------------- UI ---------------- #

    def init_ui(self):
        self.video_label = QLabel()
        self.video_label.setFixedSize(720, 540)

        self.webcam_btn = QPushButton("Webcam")
        self.video_btn = QPushButton("Video File")
        self.stop_btn = QPushButton("Stop")

        self.webcam_btn.clicked.connect(self.start_webcam)
        self.video_btn.clicked.connect(self.open_video)
        self.stop_btn.clicked.connect(self.stop)

        btns = QHBoxLayout()
        btns.addWidget(self.webcam_btn)
        btns.addWidget(self.video_btn)
        btns.addWidget(self.stop_btn)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(140)

        self.summary_label = QLabel("30s summary...")
        self.summary_label.setFrameStyle(QFrame.Shape.Box)

        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addLayout(btns)
        left.addWidget(self.log_box)
        left.addWidget(self.summary_label)

        self.bpm_view = QWebEngineView()
        self.stress_view = QWebEngineView()
        self.hrv_view = QWebEngineView()

        right = QVBoxLayout()
        right.addWidget(self.bpm_view)
        right.addWidget(self.stress_view)
        right.addWidget(self.hrv_view)

        main = QHBoxLayout()
        main.addLayout(left, 2)
        main.addLayout(right, 3)
        self.setLayout(main)

    # ---------------- STYLE ---------------- #

    def apply_theme(self):
        self.setStyleSheet("""
        QWidget { background:#111; color:#ddd; font-size:14px; }
        QPushButton {
            background:#222; border:1px solid #444;
            padding:8px; border-radius:6px;
        }
        QPushButton:hover { background:#333; }
        QTextEdit { background:#000; }
        QLabel { padding:6px; }
        """)

    # ---------------- VIDEO CONTROL ---------------- #

    def start_webcam(self):
        self.stop()
        self.vs = VideoSource(0)
        self.vs.open()
        self.start_time = time.time()
        self.frame_timer.start(1000 // FPS)
        self.graph_timer.start(GRAPH_UPDATE_INTERVAL)

    def open_video(self):
        self.stop()
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "*.mp4 *.avi")
        if path:
            self.vs = VideoSource(path)
            self.vs.open()
            self.start_time = time.time()
            self.frame_timer.start(1000 // FPS)
            self.graph_timer.start(GRAPH_UPDATE_INTERVAL)

    def stop(self):
        self.frame_timer.stop()
        self.graph_timer.stop()
        if self.vs:
            self.vs.release()
        self.vs = None

    # ---------------- FRAME LOOP ---------------- #

    def update_frame(self):
        if not self.vs:
            return

        frame = self.vs.read()
        if frame is None:
            self.stop()
            return

        landmarks, bbox = self.face.process(frame)

        if landmarks is not None:
            roi_pixels, mask = self.roi.extract(frame, landmarks)
            signal = self.rppg.update(roi_pixels)

            if signal is not None:
                self.pulse_buffer = bandpass(signal, FPS)[-240:]

            if len(self.pulse_buffer) > FPS:
                peaks = detect_peaks(np.array(self.pulse_buffer), FPS)
                if peaks is not None:
                    rr = np.diff(peaks / FPS)[-10:]
                    if len(rr) >= 2:
                        self.last_hr = compute_hr(rr)
                        self.last_rmssd = rmssd(rr)
                        self.hr_history.append(self.last_hr)
                        self.rmssd_history.append(self.last_rmssd)

                        self.last_stress = compute_stress_index(
                            self.last_hr, self.last_rmssd, list(self.hr_history)
                        )
                        self.stress_history.append(self.last_stress)

        self.display(frame)
        self.update_log()
        self.check_summary()

    # ---------------- DISPLAY ---------------- #

    def display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.width(), self.video_label.height()
        ))

    # ---------------- LOG & SUMMARY ---------------- #

    def update_log(self):
        self.log_box.setText(
            f"BPM: {self.last_hr}\n"
            f"RMSSD: {self.last_rmssd}\n"
            f"Stress Index: {self.last_stress}"
        )

    def check_summary(self):
        if time.time() - self.start_time >= SUMMARY_INTERVAL:
            if self.hr_history:
                self.summary_label.setText(
                    f"🧠 60s Summary → "
                    f"BPM: {np.mean(self.hr_history):.1f} | "
                    f"RMSSD: {np.mean(self.rmssd_history):.2f} | "
                    f"Stress: {np.mean(self.stress_history):.2f}"
                )
            self.start_time = time.time()

    # ---------------- GRAPHS ---------------- #

    def init_graphs(self):
        self.bpm_view.setHtml(PLOTLY_HTML % ("Heart Rate (BPM)", "#00ffcc"))
        self.stress_view.setHtml(PLOTLY_HTML % ("Stress Index", "#ff5555"))
        self.hrv_view.setHtml(PLOTLY_HTML % ("HRV (RMSSD)", "#ffaa00"))

    def update_graphs(self):
        self.update_plot(self.bpm_view, self.hr_history)
        self.update_plot(self.stress_view, self.stress_history)
        self.update_plot(self.hrv_view, self.rmssd_history)

    def update_plot(self, view, data):
        js = f"Plotly.update('plot', {{y: [{list(data)}]}});"
        view.page().runJavaScript(js)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StressDashboard()
    window.show()
    sys.exit(app.exec())
