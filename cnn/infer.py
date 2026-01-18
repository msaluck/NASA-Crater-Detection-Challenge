import os
import math
import cv2
import numpy as np
import torch

from cnn.model import CraterRefiner


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 96

# Safety clamps (prevent crazy corrections)
MAX_CENTER_SHIFT = 0.5   # fraction of max(a,b)
MAX_AXIS_SHIFT   = 0.5   # fraction of a/b
MAX_ANGLE_SHIFT  = 30.0  # degrees


# =========================
# UTILITIES
# =========================
def load_model(weights_path):
    model = CraterRefiner().to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def extract_crop(img, ellipse, size=CROP_SIZE):
    """
    Extract square crop centered on ellipse center.
    """
    h, w = img.shape
    cx, cy = int(ellipse["x"]), int(ellipse["y"])
    r = int(max(ellipse["a"], ellipse["b"]) * 1.5)

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    return crop


def wrap_angle(deg):
    """
    Wrap angle to [-180, 180)
    """
    while deg >= 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


# =========================
# REFINEMENT
# =========================
@torch.no_grad()
def refine_ellipses(img, ellipses, model):
    """
    img: grayscale image (H, W)
    ellipses: list of dicts with keys x,y,a,b,angle
    returns: refined ellipses (same format)
    """
    refined = []

    for e in ellipses:
        crop = extract_crop(img, e)
        if crop is None:
            refined.append(e)
            continue

        x = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
        dx, dy, da, db, dtheta = model(x)[0].cpu().numpy()

        # --- safety clamps ---
        max_r = max(e["a"], e["b"])
        dx = np.clip(dx, -MAX_CENTER_SHIFT * max_r, MAX_CENTER_SHIFT * max_r)
        dy = np.clip(dy, -MAX_CENTER_SHIFT * max_r, MAX_CENTER_SHIFT * max_r)

        da = np.clip(da, -MAX_AXIS_SHIFT * e["a"], MAX_AXIS_SHIFT * e["a"])
        db = np.clip(db, -MAX_AXIS_SHIFT * e["b"], MAX_AXIS_SHIFT * e["b"])

        dtheta = np.clip(dtheta, -MAX_ANGLE_SHIFT, MAX_ANGLE_SHIFT)

        # --- apply corrections ---
        new_e = {
            "x": e["x"] + float(dx),
            "y": e["y"] + float(dy),
            "a": max(1.0, e["a"] + float(da)),
            "b": max(1.0, e["b"] + float(db)),
            "angle": wrap_angle(e["angle"] + float(dtheta)),
            "support": e.get("support", 0)
        }

        refined.append(new_e)

    return refined
