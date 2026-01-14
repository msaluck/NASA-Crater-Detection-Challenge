import math

IMG_WIDTH = 2592
IMG_HEIGHT = 2048

def ellipse_is_valid(e):
    """
    Check ellipse validity using the SAME rules as data_combiner.py
    """
    cx = float(e["x"])
    cy = float(e["y"])
    a  = float(e["a"])
    b  = float(e["b"])
    ang = float(e["angle"])

    # Center must be inside image
    if cx < 0 or cx >= IMG_WIDTH:
        return False
    if cy < 0 or cy >= IMG_HEIGHT:
        return False

    # Compute ellipse bounding box (axis-aligned)
    alpha = math.radians(ang)

    ax = (a * math.cos(alpha)) ** 2
    bx = (b * math.sin(alpha)) ** 2
    dx = math.sqrt(ax + bx)

    ay = (a * math.sin(alpha)) ** 2
    by = (b * math.cos(alpha)) ** 2
    dy = math.sqrt(ay + by)

    # Ellipse must fully fit in image
    if cx - dx < 0 or cx + dx >= IMG_WIDTH:
        return False
    if cy - dy < 0 or cy + dy >= IMG_HEIGHT:
        return False

    # Reject very large craters (same as data_combiner)
    if 2 * (dx + dy) > 0.6 * IMG_HEIGHT:
        return False

    return True
