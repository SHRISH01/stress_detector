import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.metrics import mean_absolute_error

from core.video_source import VideoSource
from core.face import FaceLandmarkDetector
from core.roi import ROIExtractor
from rppg.green import GreenRPPG
from rppg.chrom import ChromRPPG


VIDEO_PATH = "/Users/shrish/Desktop/stress_detector/eval_data/vid.avi"
PPG_PATH = "/Users/shrish/Desktop/stress_detector/eval_data/ground_truth.txt"

FPS_RPPG = 30    
FS_PPG = 64    
LOW_HZ = 0.7
HIGH_HZ = 3.0


def bandpass(sig, fs, low=0.7, high=3.0, order=3):
    if len(sig) < fs * 2:
        return sig
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band")
    return filtfilt(b, a, sig)


def extract_rppg(video_path, method="green"):
    vs = VideoSource(video_path)
    vs.open()

    face = FaceLandmarkDetector()
    roi = ROIExtractor()

    if method == "green":
        rppg = GreenRPPG(FPS_RPPG)
    elif method == "chrom":
        rppg = ChromRPPG(FPS_RPPG)
    else:
        raise ValueError("Unknown rPPG method")

    signal = []

    while True:
        frame = vs.read()
        if frame is None:
            break

        landmarks, _ = face.process(frame)
        if landmarks is None:
            continue

        roi_pixels, _ = roi.extract(frame, landmarks)
        s = rppg.update(roi_pixels)

        if s is not None:
            signal.append(s[-1])

    vs.release()
    return np.array(signal)


def compute_hr(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs * 0.5)
    if len(peaks) < 2:
        return np.array([])
    return 60.0 / np.diff(peaks / fs)


def main():
    print("\n=== rPPG Evaluation ===\n")

    # ---------- LOAD VIDEO ----------
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video FPS: {fps:.2f}")

    # ---------- LOAD GROUND TRUTH ----------
    ppg = np.loadtxt(PPG_PATH)

    # If PPG is multi-channel, take the first row
    if ppg.ndim > 1:
        ppg = ppg[0]

    ppg = np.asarray(ppg).flatten()

    print(f"PPG samples: {len(ppg)} @ {FS_PPG} Hz")

    # ---------- EXTRACT rPPG ----------
    print("Extracting rPPG signal...")
    rppg = extract_rppg(VIDEO_PATH)
    print(f"rPPG samples: {len(rppg)} @ {fps:.2f} FPS")

    # ---------- BANDPASS ----------
    ppg_f = bandpass(ppg, FS_PPG, LOW_HZ, HIGH_HZ)
    rppg_f = bandpass(rppg, fps, LOW_HZ, HIGH_HZ)

    # ---------- RESAMPLE ----------
    N = min(len(ppg_f), len(rppg_f))
    ppg_f = resample(ppg_f, N)
    rppg_f = resample(rppg_f, N)

    # ---------- HR COMPUTATION ----------
    hr_gt = compute_hr(ppg_f, FS_PPG)
    hr_rppg = compute_hr(rppg_f, fps)

    L = min(len(hr_gt), len(hr_rppg))
    hr_gt = hr_gt[:L]
    hr_rppg = hr_rppg[:L]

    # ---------- METRICS ----------
    mae = mean_absolute_error(hr_gt, hr_rppg)
    corr = np.corrcoef(hr_gt, hr_rppg)[0, 1]

    print("\n=== RESULTS ===")
    print(f"HR MAE        : {mae:.2f} BPM")
    print(f"Correlation   : {corr:.3f}")

    # ---------- PLOTS ----------
    plt.figure(figsize=(12, 4))
    plt.plot(ppg_f, label="Ground Truth PPG")
    plt.plot(rppg_f, label="rPPG", alpha=0.7)
    plt.title("Bandpassed Signals (Aligned)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(hr_gt, label="Ground Truth HR")
    plt.plot(hr_rppg, label="rPPG HR")
    plt.ylabel("BPM")
    plt.xlabel("Beat Index")
    plt.title("Heart Rate Comparison")
    plt.legend()
    plt.show()

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
