import cv2
import numpy as np

def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5,5), 1.2)
    return img


def detect_ellipses(gray):
    h, w = gray.shape

    SCALE = 0.37  # try 0.5 first (2× speed); 0.4 if still slow 0.5 -> 0.37 (more accurate); 0.33 (even faster, less accurate)

    # edges = cv2.Canny(gray, 60, 120)
    small = cv2.resize(
        gray, 
        None, 
        fx=SCALE, 
        fy=SCALE, 
        interpolation=cv2.INTER_AREA
    )
    
    edges = cv2.Canny(small, 60, 120)

    # circles = cv2.HoughCircles(
    #     edges,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1.2,
    #     minDist=w // 10,
    #     param1=120,
    #     param2=15,
    #     minRadius=8,
    #     maxRadius=w // 4
    # )
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int((w * SCALE) // 10),
        param1=120,
        param2=22,                     # slightly stricter 18 -> 22
        minRadius=int(8 * SCALE),
        maxRadius=int((w * SCALE) // 6)  # SPEED FIX #2
    )

    ellipses = []

    if circles is None:
        return ellipses

    # circles = np.uint16(np.around(circles[0]))
    circles = np.around(circles[0]).astype(int)

    for (x, y, r) in circles:
        # =========================
        # SCALE BACK TO ORIGINAL
        # =========================
        x = int(x / SCALE)
        y = int(y / SCALE)
        r = int(r / SCALE)
        pad = int(r * 1.5)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(w, x+pad), min(h, y+pad)

        roi = edges[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not cnts:
            continue

        cnt = max(cnts, key=len)
        if len(cnt) < 40: # min number of points to fit ellipse *before 30
            continue

        cnt = cnt + np.array([[x1, y1]])

        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue

        (cx, cy), (MA, ma), angle = ellipse

        a = max(MA, ma) / 2
        b = min(MA, ma) / 2

        if b < 10 or a / b > 1.5:
            continue

        ellipses.append({
            "x": cx,
            "y": cy,
            "a": a,
            "b": b,
            "angle": angle,
            "support": len(cnt)
        })
        # =========================
        # SPEED FIX #3: EARLY STOP
        # =========================
        if len(ellipses) >= 6: # limit to max 6 craters per image before 10
            break

    return ellipses
