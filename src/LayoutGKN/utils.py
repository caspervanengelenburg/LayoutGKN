import os, matplotlib.pyplot as plt


def load_image_rplan(pid, path_rplan):
    """Loads RPLAN image as integer-valued NumPy array."""
    img = (255*plt.imread(os.path.join(path_rplan, f"{pid}.png"))[..., 1]).astype(int)
    return img