"""
Color detection using K-Means clustering on BGR frames.
Maps dominant pixel clusters to human-readable color names via HSV ranges.
No training required — pure computer vision using OpenCV + scikit-learn.

Improvements in v2:
  - Wider pink and brown HSV ranges to catch more real-world variants
  - Beige / cream / tan added as distinct color names
  - Navy blue distinguished from standard blue
  - Background filter: drops near-white and near-black clusters that
    account for >70% of the image (likely background, not garment)
  - Minimum cluster coverage raised from 25% → 20% to catch
    two-tone garments with slightly unequal color splits
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Each entry: (name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v)) in HSV space
# Ordered so that more-specific ranges come before broader ones.
_COLOR_RANGES = [
    # ── Achromatic ────────────────────────────────────────────────────────────
    ("white",  (  0,   0, 210), (180,  30, 255)),
    ("black",  (  0,   0,   0), (180, 255,  40)),
    ("gray",   (  0,   0,  41), (180,  40, 209)),

    # ── Reds (wrap-around in HSV) ─────────────────────────────────────────────
    ("red",    (  0,  70,  50), ( 10, 255, 255)),
    ("red",    (165,  70,  50), (180, 255, 255)),

    # ── Warm hues ─────────────────────────────────────────────────────────────
    ("orange", ( 11,  80,  50), ( 25, 255, 255)),
    ("yellow", ( 26,  70,  50), ( 35, 255, 255)),

    # ── Earth tones (before green so beige/brown win) ─────────────────────────
    ("beige",  (  0,  10, 180), ( 30,  60, 230)),   # cream / off-white / tan
    ("brown",  (  8,  60,  20), ( 25, 220, 160)),

    # ── Cool hues ─────────────────────────────────────────────────────────────
    ("green",  ( 36,  50,  40), ( 85, 255, 255)),
    ("navy",   ( 86,  80,  20), (130, 255,  80)),   # dark blue / navy
    ("blue",   ( 86,  50,  80), (130, 255, 255)),

    # ── Purple / pink ─────────────────────────────────────────────────────────
    ("purple", (131,  50,  40), (160, 255, 255)),
    ("pink",   (  0,  20, 150), ( 10,  70, 255)),   # light pinkish-red (low sat)
    ("pink",   (155,  30, 150), (180, 255, 255)),
]


def _bgr_to_name(bgr_pixel: np.ndarray) -> str:
    """Converts a single BGR pixel array to a human-readable color name."""
    pixel = np.uint8([[bgr_pixel]])
    hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    for name, (lo_h, lo_s, lo_v), (hi_h, hi_s, hi_v) in _COLOR_RANGES:
        if lo_h <= h <= hi_h and lo_s <= s <= hi_s and lo_v <= v <= hi_v:
            return name
    return "neutral"


def _is_background_color(bgr_pixel: np.ndarray) -> bool:
    """
    Returns True for near-white or near-black pixels that are almost
    certainly image background rather than actual garment color.
    """
    pixel = np.uint8([[bgr_pixel]])
    hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    v = int(hsv[2])
    s = int(hsv[1])
    # Very bright + low saturation → white background
    if v >= 210 and s <= 30:
        return True
    # Very dark → dark background / shadow
    if v <= 40:
        return True
    return False


def describe_colors(frame: np.ndarray, n_colors: int = 4,
                    fg_mask: np.ndarray = None) -> str:
    """
    Finds the dominant colors using K-Means clustering.

    If fg_mask is provided (a binary uint8 mask where 1 = clothing),
    color detection runs only on those pixels — ignoring background,
    shadows, hangers, and other distractors.

    Background clusters (near-white / near-black) that account for the
    majority of pixels are automatically discarded so the returned color
    reflects the actual garment, not the studio backdrop.

    Args:
        frame    : OpenCV BGR image (H x W x 3)
        n_colors : Number of K-Means clusters (default 4 for better separation)
        fg_mask  : Optional binary mask (H x W), 1 = include, 0 = ignore

    Returns:
        A string such as ``"red"``, ``"blue and white"``, or ``"navy"``
    """
    # Downsample for speed
    small = cv2.resize(frame, (80, 80))

    if fg_mask is not None:
        small_mask = cv2.resize(fg_mask, (80, 80), interpolation=cv2.INTER_NEAREST)
        pixels = small[small_mask == 1].astype(np.float32)
        if len(pixels) < n_colors * 10:   # too few masked pixels → use full image
            pixels = small.reshape(-1, 3).astype(np.float32)
    else:
        pixels = small.reshape(-1, 3).astype(np.float32)

    total = len(pixels)
    if total == 0:
        return "neutral"

    # Clamp n_colors to at most the number of unique pixels
    n_clusters = min(n_colors, total)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Sort clusters by size (most dominant first)
    counts = np.bincount(kmeans.labels_, minlength=n_clusters)
    order  = np.argsort(-counts)
    sorted_centers = kmeans.cluster_centers_[order]
    sorted_counts  = counts[order]

    # ── Filter: discard dominant background clusters ───────────────────────────
    # If the top cluster looks like a white / black background AND it covers
    # more than 70 % of pixels, skip it and promote the next cluster.
    significant = []
    for center, count in zip(sorted_centers, sorted_counts):
        ratio = count / total
        if _is_background_color(center.astype(np.uint8)) and ratio > 0.70:
            continue          # skip this background-dominant cluster
        if ratio >= 0.20:     # include clusters covering ≥20 % of the image
            significant.append((center, ratio))

    # Always include the most dominant non-background color
    if not significant:
        for center, count in zip(sorted_centers, sorted_counts):
            if not _is_background_color(center.astype(np.uint8)):
                significant = [(center, count / total)]
                break
    if not significant:
        significant = [(sorted_centers[0], sorted_counts[0] / total)]

    names = [_bgr_to_name(c.astype(np.uint8)) for c, _ in significant]

    # Deduplicate while preserving dominance order
    seen, unique = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)

    return " and ".join(unique)
