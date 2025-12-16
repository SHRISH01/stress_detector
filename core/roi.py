import cv2
import numpy as np


class ROIExtractor:
    def __init__(self):
        pass

    def extract(self, frame, landmarks):
        """
        Robust rPPG ROI using face geometry + skin masking.
        Assumes landmarks are valid and face bbox is stable.
        """
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        lm = np.array(landmarks, dtype=np.int32)

        # ---------------------------
        # FACE BOUNDING BOX (from landmarks)
        # ---------------------------
        x_min = np.min(lm[:, 0])
        x_max = np.max(lm[:, 0])
        y_min = np.min(lm[:, 1])
        y_max = np.max(lm[:, 1])

        face_w = x_max - x_min
        face_h = y_max - y_min

        # ---------------------------
        # 1️⃣ FOREHEAD ROI (RECTANGULAR)
        # ---------------------------
        fh_top = int(y_min + 0.05 * face_h)
        fh_bottom = int(y_min + 0.30 * face_h)

        fh_left = int(x_min + 0.25 * face_w)
        fh_right = int(x_min + 0.75 * face_w)

        cv2.rectangle(
            mask,
            (fh_left, fh_top),
            (fh_right, fh_bottom),
            255,
            -1
        )

        # ---------------------------
        # 2️⃣ LEFT CHEEK ROI
        # ---------------------------
        lc_top = int(y_min + 0.45 * face_h)
        lc_bottom = int(y_min + 0.75 * face_h)

        lc_left = int(x_min + 0.10 * face_w)
        lc_right = int(x_min + 0.35 * face_w)

        cv2.rectangle(
            mask,
            (lc_left, lc_top),
            (lc_right, lc_bottom),
            255,
            -1
        )

        # ---------------------------
        # 3️⃣ RIGHT CHEEK ROI
        # ---------------------------
        rc_top = lc_top
        rc_bottom = lc_bottom

        rc_left = int(x_min + 0.65 * face_w)
        rc_right = int(x_min + 0.90 * face_w)

        cv2.rectangle(
            mask,
            (rc_left, rc_top),
            (rc_right, rc_bottom),
            255,
            -1
        )

        # ---------------------------
        # 4️⃣ REMOVE EYES & MOUTH (LANDMARK MASKING)
        # ---------------------------
        # Eyes
        for eye_ids in [range(36, 42), range(42, 48)]:
            eye_pts = lm[list(eye_ids)]
            cv2.fillConvexPoly(mask, eye_pts, 0)

        # Mouth
        mouth_pts = lm[48:68]
        cv2.fillConvexPoly(mask, mouth_pts, 0)

        # ---------------------------
        # 5️⃣ SKIN MASK (YCrCb)
        # ---------------------------
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(
            ycrcb,
            (0, 133, 77),
            (255, 173, 127)
        )

        final_mask = cv2.bitwise_and(mask, skin)
        roi_pixels = frame[final_mask == 255]

        return roi_pixels, final_mask
