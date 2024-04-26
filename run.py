from Depth_Estimation import MiDaS, MonoDepth2, DPT, ZoeDepth
from processing import Processor
from segmentation import Segmentator
from data_handler import InputHandler
import torch
import argparse
import os
import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
mpl.use("pgf")

# check if cuda is available otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the ArgumentParser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument(
    "-i",
    "--input",
    help="Path to image, input directory or video file",
    required=False,
    default="input/images/",
)
parser.add_argument(
    "-o",
    "--output",
    help="Path to the output folder",
    required=False,
    default="output/images",
)
parser.add_argument(
    "-et",
    "--estimator_type",
    help="Estimator model for the depth estimation",
    choices=["Mono2", "MiDaS", "ZoeDepth", "DPT", None],
    required=False,
    default="Mono2",
)
parser.add_argument(
    "-s", "--seg", help="Also segment the image", required=False, action="store_true"
)

# parse the arguments
args = parser.parse_args()

segmentator = None
depth_estimator = None
midas_model_list = ["dpt_swin2_tiny_256", "midas_v21_384", "dpt_beit_large_384"]
dpt_model_list = ["dpt_hybrid", "dpt_hybrid_kitti", "dpt_large"]
zoe_model_list = ["ZoeD_N", "ZoeD_K", "ZoeD_NK"]

# if segmentator flag is set then load the segmentator
if args.seg:
    segmentator = Segmentator(model_path="DeepLabV3Plus_resnet50.onnx", device=device)

# if estimator_type is set then load the estimator accordingly depending which type should be used
if args.estimator_type == "Mono2":
    model_type = "mono+stereo_640x192"
    depth_estimator = MonoDepth2(device=device, model_name=model_type)

elif args.estimator_type == "MiDaS":
    model_type = midas_model_list[0]
    depth_estimator = MiDaS(device=device, model_type=model_type, optimize=False)

elif args.estimator_type == "DPT":
    model_type = dpt_model_list[1]
    depth_estimator = DPT(device=device, model_type=model_type)

elif args.estimator_type == "ZoeDepth":
    model_type = "ZoeD_NK"
    depth_estimator = ZoeDepth(device=device, model_type=model_type)

print("Using:", args.estimator_type + "_" + model_type)

# initialize the InputHandler accordingly with input/output arguments
ih = InputHandler(
    input_path=args.input,
    output_path_depth=os.path.join(args.output, args.estimator_type, model_type),
    output_path_seg=args.output,
)

# initialize the Processor accordingly with segmentator, depth estimator and estimator type
processor = Processor(
    ih=ih,
    segmentator=segmentator,
    depth_estimator=depth_estimator,
    estimator_type=args.estimator_type + "_" + model_type,
)

# process the input data
processor.process()

# store/plot the statistical resutls
processor.return_timer_stats()

print("Terminated--")
