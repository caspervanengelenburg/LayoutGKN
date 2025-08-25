# Categories RPLAN original
CAT_RPLAN_ORIG = {
    0: "living room",
    1: "master room",
    2: "kitchen",
    3: "bathroom",
    4: "dining room",
    5: "child room",
    6: "study room",
    7: "second room",
    8: "guest room",
    9: "balcony",
    10: "entrance",
    11: "storage",
    12: "walk-in",
    13: "external area",
    14: "exterior wall",
    15: "front door",
    16: "interior wall",
    17: "interior door"
}

# Old to new mapping
CAT_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 6,
    10: 7,
    11: 5,
    12: 5
}


# Categories RPLAN new (ie, more universal)
CAT_RPLAN = {
    0: "living room",
    1: "bedroom",
    2: "kitchen",
    3: "bathroom",
    4: "dining room",
    5: "storage",
    6: "balcony",
    7: "corridor"
}

# Old to new mapping
CAT_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 6,
    10: 7,
    11: 5,
    12: 5
}


ROOM_COLORS = [
    '#e6550d',  # living room
    '#1f77b4',  # bedroom
    '#fd8d3c',  # kitchen
    '#6b6ecf',  # bathroom
    '#fdae6b',  # dining
    '#5254a3',  # store room
    '#2ca02c',  # balcony
    '#fdd0a2'   # corridor
]