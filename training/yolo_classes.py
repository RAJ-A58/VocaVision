"""
VocaVision — YOLO Class Definitions
All 20 target classes: 10 Food + 10 Clothing

These are the YOLO model's output class IDs.
Open Images V7 class names are mapped here for dataset download.

NOTE: OI_FOOD_MAP and OI_CLOTHING_MAP must match what was actually
downloaded from Open Images — do not change without re-downloading.
"""

# -- Our 20 target classes ----------------------------------------------------
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

# -- Open Images V7 -> VocaVision class name mapping --------------------------
# Keys are EXACT Open Images display class labels (verified valid).
# Values are our target class names.
#
# FOOD — classes that were actually downloaded (9 valid; "Noodle" was invalid)
OI_FOOD_MAP = {
    "Pizza":        "pizza",
    "Hamburger":    "hamburger",
    "Hot dog":      "hot_dog",
    "Ice cream":    "ice_cream",
    "Sushi":        "sushi",
    "Waffle":       "waffles",
    "French fries": "fried_rice",
    "Cake":         "apple_pie",
    "Pretzel":      "samosa",    # downloaded; maps to samosa slot
    # "Noodle" was INVALID (not in Open Images) -- ramen slot will be sparse
}

# CLOTHING — classes that were actually downloaded ("T-shirt" and "Shoe" were invalid)
OI_CLOTHING_MAP = {
    "Trousers":   "trouser",
    "Dress":      "dress",
    "Coat":       "coat",
    "Shorts":     "pullover",    # downloaded; maps to pullover slot
    "High heels": "sandal",      # downloaded; maps to sandal slot
    "Handbag":    "bag",
    "Jacket":     "shirt",
    "Boot":       "ankle_boot",
    # "T-shirt" was INVALID; t_shirt slot will be sparse
    # "Shoe"    was INVALID; sneaker slot will be sparse
}

OI_ALL_MAP = {**OI_FOOD_MAP, **OI_CLOTHING_MAP}

# Which Open Images classes to download
OI_FOOD_CLASSES     = list(OI_FOOD_MAP.keys())
OI_CLOTHING_CLASSES = list(OI_CLOTHING_MAP.keys())
OI_ALL_OI_CLASSES   = OI_FOOD_CLASSES + OI_CLOTHING_CLASSES
