import os
import cv2
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def load_image(img_root, image_id):
    """
    image_id: altitudeXX/longitudeYY/orientationZZ_lightWW
    """
    path = os.path.join(
        img_root,
        image_id.split("/")[0],
        image_id.split("/")[1],
        image_id.split("/")[2] + ".png"
    )
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def ellipse_distance(e1, e2):
    """Simple center distance"""
    return math.hypot(e1["x"] - e2["x"], e1["y"] - e2["y"])


def extract_crop(img, ellipse, size=96):
    """
    Extract square crop centered on ellipse center
    """
    cx, cy = int(ellipse["x"]), int(ellipse["y"])
    r = int(max(ellipse["a"], ellipse["b"]) * 1.5)

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(img.shape[1], cx + r)
    y2 = min(img.shape[0], cy + r)

    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    return crop


class CraterDataset(Dataset):
    def __init__(
        self,
        gt_csv,
        cv_csv,
        img_root,
        max_center_dist_ratio=1.0,
        size=96
    ):
        """
        gt_csv: train-gt.csv
        cv_csv: output/solution.csv
        img_root: nasa-craters-data/train
        """
        self.img_root = img_root
        self.size = size
        self.samples = []

        gt = pd.read_csv(gt_csv)
        cv = pd.read_csv(cv_csv)

        # group by image
        gt_groups = gt.groupby("inputImage")
        cv_groups = cv.groupby("inputImage")

        for image_id in gt_groups.groups:
            if image_id not in cv_groups.groups:
                continue

            gt_rows = gt_groups.get_group(image_id)
            cv_rows = cv_groups.get_group(image_id)

            # skip empty GT images
            if (gt_rows["ellipseSemimajor(px)"] < 0).all():
                continue

            img = load_image(img_root, image_id)
            if img is None:
                continue

            # build GT ellipses
            gt_ellipses = []
            for _, r in gt_rows.iterrows():
                if r["ellipseSemimajor(px)"] < 0:
                    continue
                gt_ellipses.append({
                    "x": r["ellipseCenterX(px)"],
                    "y": r["ellipseCenterY(px)"],
                    "a": r["ellipseSemimajor(px)"],
                    "b": r["ellipseSemiminor(px)"],
                    "angle": r["ellipseRotation(deg)"]
                })

            # build CV ellipses
            cv_ellipses = []
            for _, r in cv_rows.iterrows():
                if r["ellipseSemimajor(px)"] < 0:
                    continue
                cv_ellipses.append({
                    "x": r["ellipseCenterX(px)"],
                    "y": r["ellipseCenterY(px)"],
                    "a": r["ellipseSemimajor(px)"],
                    "b": r["ellipseSemiminor(px)"],
                    "angle": r["ellipseRotation(deg)"]
                })

            # match CV → GT
            for cv_e in cv_ellipses:
                best_gt = None
                best_dist = 1e9

                for gt_e in gt_ellipses:
                    d = ellipse_distance(cv_e, gt_e)
                    if d < best_dist:
                        best_dist = d
                        best_gt = gt_e

                if best_gt is None:
                    continue

                # loose acceptance rule
                if best_dist > max(cv_e["a"], cv_e["b"]) * max_center_dist_ratio:
                    continue

                crop = extract_crop(img, cv_e, self.size)

                dx = best_gt["x"] - cv_e["x"]
                dy = best_gt["y"] - cv_e["y"]
                da = best_gt["a"] - cv_e["a"]
                db = best_gt["b"] - cv_e["b"]
                dtheta = best_gt["angle"] - cv_e["angle"]

                target = np.array([dx, dy, da, db, dtheta], dtype=np.float32)

                self.samples.append((crop, target))

        print(f"[CraterDataset] built {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, target = self.samples[idx]
        img = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        target = torch.from_numpy(target)
        return img, target
