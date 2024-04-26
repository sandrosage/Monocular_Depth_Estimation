from .DPT.dpt.models import DPTDepthModel
from .DPT.dpt.midas_net import MidasNet_large
from .DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from .monodepth2.utils import download_model_if_doesnt_exist
from .monodepth2 import networks

from torchvision.transforms import Compose
import cv2
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

zoe_model_types = ["ZoeD_N", "ZoeD_K", "ZoeD_NK"]

dpt_default_models = {
    "midas_v21": "weights/midas_v21-f6b98070.pt",
    "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
    "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
}


def load_dpt_model(device, model_type, model_path, optimize):
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert False, f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize is True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    return model, transform, net_w, net_h


def load_mono2_model(model_name, device):
    path = "Depth_Estimation/monodepth2/models"
    download_model_if_doesnt_exist(model_name, path)
    # extra for this application change the path
    model_path = os.path.join(path, model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4)
    )

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    return encoder, depth_decoder, feed_width, feed_height


def load_pil(img_path):
    img_rgb = Image.open(img_path).convert("RGB")
    return img_rgb


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(float) / 256.0
    mask = depth_png != 0
    return depth, mask


def store_depth(depth, path, format="png"):
    """
    Storing of depth maps in matplotlib.pyplot

    Args:
        - depth: depth map in numpy format
        - path: output path
        - format (default="png"): format of image -> "pgf" for latex support
    """
    plt.imshow(depth)
    plt.colorbar(orientation="horizontal")
    if format == "png":
        path = path + ".png"
        plt.savefig(path, dpi=1000, format=format, bbox_inches="tight")
    elif format == "pgf":
        path = path + ".pgf"
        plt.savefig(path, backend="pgf", dpi=1000)
    plt.clf()
