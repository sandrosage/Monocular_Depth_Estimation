from Depth_Estimation import MiDaS, MonoDepth2, DPT, ZoeDepth
from processing import Processor
from segmentation import Segmentator
from data_handler import InputHandler
import torch 
import argparse
import os

# check if cuda is available otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize the ArgumentParser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("-i","--input", help="Path to image, input directory or video file", required=False, default="depth_selection/val_selection_cropped/image/2011_10_03_drive_0047_sync_image_0000000791_image_03.png")
parser.add_argument("-o", "--output", help="Path to the output folder", required=False, default="output/images")
parser.add_argument("-et", "--estimator_type", help="Estimator model for the depth estimation", choices=["Mono2","MiDaS", "ZoeDepth", "DPT", None], required=False, default="Mono2")
parser.add_argument("-emt", "--estimator_model_type", help="which kind of model type should be used", choices=["small", "medium", "large", None], required=False, default="medium")
parser.add_argument("-s", "--seg", help="Also segment the image", type=bool, required=False)

# parse the arguments
args = parser.parse_args()

segmentator = None
depth_estimator = None

# if segmentator flag is set then load the segmentator
if args.seg:
    segmentator = Segmentator(model_path="DeepLabV3Plus_resnet50.onnx", device=device)

# if estimator_type is set then load the estimator accordingly depending which type should be used
if args.estimator_type == "Mono2":
    depth_estimator = MonoDepth2(device=device, model_name="mono+stereo_640x192")

elif args.estimator_type == "MiDaS":
    depth_estimator = MiDaS(device=device, model_type="midas_v21_384", optimize=False)

elif args.estimator_type == "DPT":
    depth_estimator = DPT(device=device, model_type="dpt_hybrid_kitti")

elif args.estimator_type == "ZoeDepth":
    depth_estimator = ZoeDepth(device=device, model_type="ZoeD_K")

print("Using:", args.estimator_type)

# initialize the InputHandler accordingly with input/output arguments
ih = InputHandler(input_path=args.input, output_path_depth=os.path.join(args.output, args.estimator_type), output_path_seg=args.output)

# initialize the Processor accordingly with segmentator, depth estimator and estimator type
processor = Processor(ih=ih, segmentator=segmentator, depth_estimator=depth_estimator, estimator_type=args.estimator_type)

# process the input data
processor.process()

# store/plot the statistical resutls 
processor.return_timer_stats()

print("Terminated--")