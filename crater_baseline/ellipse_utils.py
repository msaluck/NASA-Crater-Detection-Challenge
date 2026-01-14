def suppress_overlaps(ellipses, max_keep=6):
    ellipses = sorted(ellipses, key=lambda e: e["support"], reverse=True)

    kept = []

    for e in ellipses:
        ok = True
        for k in kept:
            dist = ((e["x"] - k["x"])**2 + (e["y"] - k["y"])**2) ** 0.5
            if dist < min(e["b"], k["b"]):
                ok = False
                break
        if ok:
            kept.append(e)
        if len(kept) >= max_keep:
            break

    return kept
