# useful functions for input/output handling + necessary output for latex
import matplotlib.pyplot as plt


def store_depth(depth, path, format="png", flag=None):
    """
    Storing of depth maps in matplotlib.pyplot

    Args:
        - depth: depth map in numpy format
        - path: output path
        - format (default="png"): format of image -> "pgf" for latex support
        - flag: text to put in image (usually model type )
    """
    plt.imshow(depth)
    plt.colorbar(orientation="horizontal")
    if format == "png":
        new_path = path + ".png"
        if flag is not None:
            new_path = path + "_" + flag + ".png"
            plt.text(
                20,
                depth.shape[0] - 20,
                flag,
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        plt.savefig(new_path, dpi=1000, format=format, bbox_inches="tight")
    elif format == "pgf":
        path = path + ".pgf"
        plt.savefig(path, backend="pgf", dpi=1000)
    plt.clf()
