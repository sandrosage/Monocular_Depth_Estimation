# useful functions for input/output handling + necessary output for latex
import matplotlib.pyplot as plt

def store_depth(depth, path, format="png"):
    """
    Storing of depth maps in matplotlib.pyplot

    Args:
        - depth: depth map in numpy format
        - path: output path 
        - format (default="png"): format of image -> "pgf" for latex support
    """
    if format == "png":
        path = path + ".png"
    elif format == "pgf":
        path = path + ".pgf"
    plt.imshow(depth)
    plt.colorbar(orientation="horizontal")
    plt.savefig(path, dpi=600, format=format)
    plt.clf()