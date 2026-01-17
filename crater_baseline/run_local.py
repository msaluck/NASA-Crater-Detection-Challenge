import cv2
import os
import time
from geometry_filter import ellipse_is_valid
from detect import preprocess, detect_ellipses
from ellipse_utils import suppress_overlaps
from save import save_csv
from tqdm import tqdm
import subprocess

IMG_DIR = "../nasa-craters-data/train/"
TRUTH_DIR = "../nasa-craters-data/train-gt.csv"
OUT = "output/solution.csv"
start_time = time.time()

results = {}

# collect all image paths first
image_paths = []

for root, _, files in os.walk(IMG_DIR):
    # ❌ skip ground-truth folders
    if os.path.basename(root) == "truth":
        continue

    for f in files:
        if f.endswith(".png"):
            image_paths.append(os.path.join(root, f))

print(f"Found {len(image_paths)} images")


for img_path in tqdm(image_paths, desc="Processing images"):
    img = cv2.imread(img_path, 0)
    if img is None:
        continue

    proc = preprocess(img)
    ellipses = detect_ellipses(proc)

    # 1️⃣ geometry filter FIRST
    ellipses = [e for e in ellipses if ellipse_is_valid(e)]

    # 2️⃣ relax size threshold
    ellipses = [e for e in ellipses if e["b"] >= 7]

    # 3️⃣ overlap suppression LAST
    ellipses = suppress_overlaps(ellipses)

    # 4️⃣ scorer-aware cap
    ellipses = ellipses[:8]

    norm = os.path.normpath(img_path)
    parts = [p for p in norm.split(os.sep) if p]

    # [..., 'train', 'altitudeXXXX', 'longitudeXX', 'orientation_lightXX.png']

    image_id = os.path.join(
        parts[-3],          # altitudeXXXX
        parts[-2],          # longitudeXX
        parts[-1][:-4]      # orientation_lightXX
    )

    results[image_id] = ellipses

    if len(results) % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"{len(results)} images in {elapsed:.1f}s")


save_csv(results, OUT)

subprocess.run([
    "python", "scorer.py",
    "--pred", OUT,
    "--truth", TRUTH_DIR,
    "--out_dir", "output/"
])
