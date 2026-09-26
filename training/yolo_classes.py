"""
VocaVision — YOLO Class Definitions
All 20 target classes: 10 Food + 10 Clothing

These are the YOLO model's output class IDs.
Open Images V7 class names are mapped here for dataset download.
"""

# ── Our 20 target classes ─────────────────────────────────────────────────────
FOOD_CLASSES = [
    "apple_pie",   # 0
    "fried_rice",  # 1
    "hamburger",   # 2
    "hot_dog",     # 3
    "ice_cream",   # 4
    "pizza",       # 5
    "ramen",       # 6
    "samosa",      # 7
    "sushi",       # 8
    "waffles",     # 9
]

CLOTHING_CLASSES = [
    "t_shirt",    # 10
    "trouser",    # 11
    "pullover",   # 12
    "dress",      # 13
    "coat",       # 14
    "sandal",     # 15
    "shirt",      # 16
    "sneaker",    # 17
    "bag",        # 18
    "ankle_boot", # 19
]

ALL_CLASSES = FOOD_CLASSES + CLOTHING_CLASSES

# ── Open Images V7 → VocaVision class name mapping ──────────────────────────
# Keys are exact Open Images class labels; values are our target class names
OI_FOOD_MAP = {
    "Pizza":        "pizza",
    "Hamburger":    "hamburger",
    "Hot dog":      "hot_dog",
    "Ice cream":    "ice_cream",
    "Sushi":        "sushi",
    "Waffle":       "waffles",
    "Noodle":       "ramen",
    "French fries": "fried_rice",
    "Cake":         "apple_pie",
    "Pretzel":      "samosa",   # visual substitute (round + fried dough)
}

OI_CLOTHING_MAP = {
    "T-shirt":   "t_shirt",
    "Trousers":  "trouser",
    "Dress":     "dress",
    "Coat":      "coat",
    "Shorts":    "pullover",  # catch-all upper body
    "Shoe":      "sneaker",
    "High heels":"sandal",
    "Handbag":   "bag",
    "Jacket":    "shirt",
    "Boot":      "ankle_boot",
}

OI_ALL_MAP = {**OI_FOOD_MAP, **OI_CLOTHING_MAP}

# Which Open Images classes to download
OI_FOOD_CLASSES    = list(OI_FOOD_MAP.keys())
OI_CLOTHING_CLASSES = list(OI_CLOTHING_MAP.keys())
OI_ALL_OI_CLASSES  = OI_FOOD_CLASSES + OI_CLOTHING_CLASSES
