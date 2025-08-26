import os, matplotlib.pyplot as plt
import pickle
from shapely.geometry import Polygon

def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def load_image_rplan(pid, path_rplan):
    """Loads RPLAN image as integer-valued NumPy array."""
    img = (255*plt.imread(os.path.join(path_rplan, f"{pid}.png"))[..., 1]).astype(int)
    return img


def minmax_normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))