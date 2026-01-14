import pandas as pd

def save_csv(results, out_path):
    rows = []

    for image_id, ellipses in results.items():

        # Case: NO crater detected (mandatory format)
        if not ellipses:
            rows.append({
                "inputImage": image_id,
                "ellipseCenterX(px)": 0,
                "ellipseCenterY(px)": 0,
                "ellipseSemimajor(px)": -1,
                "ellipseSemiminor(px)": -1,
                "ellipseRotation(deg)": 0
            })
            continue

        # Case: one or more detected craters
        for e in ellipses:
            rows.append({
                "inputImage": image_id,
                "ellipseCenterX(px)": float(e["x"]),
                "ellipseCenterY(px)": float(e["y"]),
                "ellipseSemimajor(px)": float(e["a"]),
                "ellipseSemiminor(px)": float(e["b"]),
                "ellipseRotation(deg)": float(e["angle"])
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
