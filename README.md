# Real-Time Stress Detection Using rPPG and Facial Analysis

## Overview

This project implements a **real-time, contactless stress monitoring system** using a standard webcam or video input.  
It estimates **Heart Rate (HR)**, **Heart Rate Variability (HRV)**, and a derived **Stress Index** by extracting physiological signals from facial video using **remote Photoplethysmography (rPPG)**.

The system is designed to be:
- Real-time and non-invasive
- Robust to facial expressions and moderate motion
- Visually interpretable through live graphs and ROI overlays
- Modular and extensible for research or deployment use cases

---

## System Pipeline

1. Face detection and landmark extraction  
2. Robust ROI extraction (forehead + cheeks)  
3. rPPG signal extraction from skin pixels  
4. Signal filtering and peak detection  
5. HR and HRV computation  
6. Stress index estimation  
7. Real-time visualization (video, graphs, logs)

---

## Logic: Stress Index Formulation

### Physiological Motivation

Stress is not defined by heart rate alone.  
From autonomic nervous system physiology:

- **Sympathetic activation** increases heart rate
- **Parasympathetic withdrawal** reduces HRV (especially RMSSD)
- Stress corresponds to **high HR + low HRV**, not simply high HR

For example:
- Exercise or laughter → HR increases but HRV often remains high
- Psychological stress → HR may increase modestly, but HRV drops significantly

Therefore, HR must be treated as **context**, while HRV acts as the **primary stress marker**.

---

### Signals Used

The stress index combines three signals:

1. **Relative Heart Rate Change**
   - Measures deviation from a personal baseline
   - Avoids penalizing naturally higher resting heart rates

2. **RMSSD (HRV)**
   - Primary indicator of parasympathetic activity
   - Lower RMSSD → higher stress

3. **Short-term HR Trend**
   - Captures sustained arousal rather than momentary spikes

---

### Final Stress Index Formula

The stress index is computed as a weighted combination:

Stress = 0.25 × Relative HR Component + 0.35 × (1 − Normalized RMSSD) + 0.15 × HR Trend Component


A **parasympathetic override** is applied:
- When RMSSD is high, stress is explicitly suppressed
- This prevents false positives during laughter, speaking, or light movement

The final output is scaled to a **0–100 range** for interpretability.

---

### Why This Design Works

- HR alone does not dominate the stress signal
- HRV has the strongest influence, as supported by clinical literature
- Relative normalization adapts to individual physiology
- The model avoids labeling physical activity or positive arousal as stress

This approach produces a smoother, more physiologically plausible stress signal than HR-dominant methods.

---

## Model Choice

### Face Detection and Landmark Extraction

A pre-trained facial landmark detector is used to:
- Reliably localize facial geometry
- Identify stable anatomical regions
- Exclude non-informative areas (eyes, mouth, jaw)

**Why this choice**
- Mature, well-tested
- Low latency
- Works in real time on CPU
- No training data required

Face detection accuracy is critical, but face recognition or identity inference is not performed.

---

### ROI Design (Critical Component)

The rPPG signal quality depends heavily on ROI selection.

Instead of using the full face or landmark polygons, the system uses:
- **Forehead (primary ROI)**  
- **Left and right cheeks (secondary ROIs)**  

These regions are defined:
- Relative to the face bounding box
- Independent of eyebrow or mouth motion
- With explicit exclusion of eyes and lips

A **skin color mask (YCrCb space)** is applied to ensure that only valid skin pixels contribute to the signal.

This ROI design is consistent with rPPG literature and significantly improves signal stability.

---

### rPPG Extraction Method

The current implementation uses:
- **Green-channel rPPG** as a baseline method

Why start with Green:
- Simple and interpretable
- Sensitive to ROI quality (useful for debugging)
- Computationally lightweight

The system architecture allows easy substitution with:
- CHROM
- POS
- PCA/ICA-based methods

---

## Signal Processing Choices

- Bandpass filtering in the **0.7–3.0 Hz** range  
  (corresponds to ~42–180 BPM)
- Peak detection on filtered signal
- RR intervals derived from peak timing
- RMSSD computed from short RR windows

These parameters balance responsiveness with noise suppression.

---

## Trade-offs and Performance Considerations

### Latency vs Accuracy

**Challenges**
- Face landmark detection
- rPPG extraction
- Real-time graph updates

**Design decisions**
- Lightweight classical methods instead of deep learning
- Small rolling buffers instead of long windows
- Throttled graph updates (1 Hz) instead of per-frame redraws

This ensures smooth real-time performance without sacrificing signal quality.

---

### GUI Responsiveness

To avoid UI freezing:
- Video capture and processing are driven by a non-blocking timer
- Graph rendering is decoupled from frame processing
- Plot updates reuse existing canvases instead of reloading content

This keeps latency low even on modest hardware.

---

### Threading Considerations

Currently:
- All processing runs in a controlled event loop
- No blocking operations are used

The architecture supports:
- Moving rPPG extraction or face processing to a separate thread if required
- This was intentionally deferred to keep the system debuggable during development

---

## Visualization and Interpretability

The interface provides:
- Live video with ROI overlay (for validation)
- Time-series graphs for:
  - Heart Rate
  - Stress Index
  - RMSSD
- Numerical status log
- Rolling 60-second summary of physiological state

This design makes the system suitable for:
- Demonstrations
- Research analysis
- Debugging and validation

---

## Limitations

- rPPG is sensitive to lighting conditions
- Large head movements can degrade signal quality
- Stress index is a proxy, not a clinical diagnosis
- Performance depends on camera quality and frame rate

These limitations are inherent to camera-based physiological sensing.

---

## Future Improvements

- POS or CHROM rPPG as default
- Signal Quality Index (SQI)
- Adaptive calibration phase
- Multi-face handling
- Session export and reporting
- Optional deep learning rPPG models

---