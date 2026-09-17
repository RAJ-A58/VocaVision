"""
Color detection using K-Means clustering on BGR frames.
Maps dominant pixel clusters to human-readable color names via HSV ranges.
No training required — pure computer vision using OpenCV + scikit-learn.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Each entry: (name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v)) in HSV space
_COLOR_RANGES = [
    ("red",    (  0,  80,  50), ( 10, 255, 255)),
    ("red",    (165,  80,  50), (180, 255, 255)),  # red wraps in HSV
    ("orange", ( 11,  80,  50), ( 25, 255, 255)),
    ("yellow", ( 26,  80,  50), ( 35, 255, 255)),
    ("green",  ( 36,  50,  50), ( 85, 255, 255)),
    ("blue",   ( 86,  50,  50), (130, 255, 255)),
    ("purple", (131,  50,  50), (155, 255, 255)),
    ("pink",   (156,  40, 150), (164, 255, 255)),
    ("white",  (  0,   0, 200), (180,  30, 255)),
    ("black",  (  0,   0,   0), (180, 255,  40)),
    ("gray",   (  0,   0,  41), (180,  30, 199)),
    ("brown",  ( 10,  60,  20), ( 20, 200, 130)),
]


def _bgr_to_name(bgr_pixel: np.ndarray) -> str:
    """Converts a single BGR pixel array to a human-readable color name."""
    pixel = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    for name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) in _COLOR_RANGES:
        if lo_h <= h <= hi_h and lo_s <= s <= hi_s and lo_v <= v <= hi_v:
            return name
    return "neutral"


def describe_colors(frame: np.ndarray, n_colors: int = 2) -> str:
    """
    Finds the dominant colors in a BGR frame using K-Means clustering,
    then maps each cluster center to a human-readable name.

    Args:
        frame   : OpenCV BGR image (H x W x 3 numpy array)
        n_colors: Number of dominant color clusters to extract

    Returns:
        A string such as "blue and white" or "dark red"
    """
    # Downsample for speed — color detection does not need full resolution
    small = cv2.resize(frame, (80, 80))
    pixels = small.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Sort centers by cluster size so most-dominant color comes first
    counts = np.bincount(kmeans.labels_)
    sorted_centers = kmeans.cluster_centers_[np.argsort(-counts)]

    names = [_bgr_to_name(c.astype(np.uint8)) for c in sorted_centers]

    # Deduplicate while preserving dominance order
    seen, unique = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    return " and ".join(unique)
