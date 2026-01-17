import cv2
import numpy as np

def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5,5), 1.2)
    return img


def detect_ellipses(gray):
    h, w = gray.shape
    SCALE = 0.4

    # --- small image for Hough ---
    small = cv2.resize(
        gray,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_AREA
    )
    edges_small = cv2.Canny(small, 60, 120)

    # --- full-res edges for ellipse fitting ---
    edges_full = cv2.Canny(gray, 60, 120)

    circles = cv2.HoughCircles(
        edges_small,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int((w * SCALE) // 10),
        param1=120,
        param2=20,
        minRadius=int(8 * SCALE),
        maxRadius=int((w * SCALE) // 6)
    )

    ellipses = []
    if circles is None:
        return ellipses

    circles = np.around(circles[0]).astype(int)

    for (x, y, r) in circles:
        # scale back
        x = int(x / SCALE)
        y = int(y / SCALE)
        r = int(r / SCALE)

        pad = int(r * 1.5)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + pad), min(h, y + pad)

        roi = edges_full[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        points = []
        for c in cnts:
            if len(c) >= 20:
                points.append(c)

        if not points:
            continue

        points = np.vstack(points)
        points = points + np.array([[x1, y1]])

        try:
            ellipse = cv2.fitEllipse(points)
        except:
            continue

        (cx, cy), (MA, ma), angle = ellipse
        a = max(MA, ma) / 2
        b = min(MA, ma) / 2

        if b < 7 or a / b > 1.5:
            continue

        ellipses.append({
            "x": cx,
            "y": cy,
            "a": a,
            "b": b,
            "angle": angle,
            "support": len(points)
        })

        if len(ellipses) >= 8:
            break

    return ellipses

