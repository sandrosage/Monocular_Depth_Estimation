from .MiDaS.midas.model_loader import load_model, default_models
from .MiDaS.utils import read_image
from .utils import load_dpt_model, load_mono2_model, zoe_model_types, load_pil, dpt_default_models, depth_read, store_depth
from .monodepth2.layers import disp_to_depth
import torch
import numpy as np
import cv2
from abc import ABC
from torchvision import transforms
from PIL.Image import Resampling

# MONODEPTH2
# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.

STEREO_SCALE_FACTOR = 5.4

first_execution = True


class DepthEstimator(ABC):
    pass


class MiDaS(DepthEstimator):
    def __init__(self, device, model_type, optimize=False, to_metric=True):
        self.first_aligning_run = True
        self.disparity_cap = 1.0 / 80
        model_path = default_models[model_type]
        model_path = "Depth_Estimation/MiDaS/" + model_path
        self.model_type = model_type
        self.device = device
        self.optimize = optimize
        self.model, self.transform, self.net_w, self.net_h = load_model(device, model_path, model_type, optimize, height=None, square=False)
        if to_metric:
            self.first_aligning_run = True
            self.scale, self.translation = self.compute_scale_and_shift()
    
    def _forward(self, sample, target_size):
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction
    
    def predict(self, img_path, use_camera=False):
        img_rgb = read_image(img_path)
        image = self.transform({"image": img_rgb})["image"]
        with torch.no_grad():
            global first_execution

            target_size = img_rgb.shape[1::-1]

            if "openvino" in self.model_type:
                if first_execution or not use_camera:
                    print(f"    Input resized to {self.net_w}x{self.net_h} before entering the encoder")
                    first_execution = False

                sample = [np.reshape(image, (1, 3, self.net_w, self.net_h))]
                prediction = self.model(sample)[self.model.output(0)][0]
                prediction = cv2.resize(prediction, dsize=target_size,
                                        interpolation=cv2.INTER_CUBIC)
            else:
                sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

                if self.optimize and self.device == torch.device("cuda"):
                    if first_execution:
                        print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                            "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                            "  half-floats.")
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()

                if first_execution or not use_camera:
                    height, width = sample.shape[2:]
                    print(f"    Input resized to {width}x{height} before entering the encoder")
                    first_execution = False

                prediction = self._forward(sample, target_size)
                if not self.first_aligning_run:
                    print("hello")
                    prediction[prediction < 0] = 0
                    prediction_aligned = self.scale * prediction + self.translation
                    prediction_aligned[prediction_aligned < self.disparity_cap] = self.disparity_cap
                    prediction = 1.0 / prediction_aligned
        return prediction
    
    def compute_scale_and_shift(self):
        """
        Computes and sets the scale and shift for the aligned prediction
        -> is needed when performing metric depth instead of relative depth
        """
        # 1. Load the ground truth data from the kitti dataset
        depth_map, mask = depth_read("depth_selection/val_selection_cropped/groundtruth_depth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000791_image_03.png")
        # 2. Transform the absolute grouth truth depth into disparity
        target_disparity = np.zeros_like(depth_map)
        target_disparity[mask == 1] = 1.0 / depth_map[mask == 1]
        # visualize it for paper
        store_depth(target_disparity, "midas_target_disparity")
        store_depth(depth_map, "midas_detph_map")
        # 3. Flatten the depth and disparity for alignment
        depth_map = depth_map.flatten()
        target_disparity = target_disparity.flatten()
        # 4. Get the not yet aligned prediction of the model
        prediction = self.predict("depth_selection/val_selection_cropped/image/2011_10_03_drive_0047_sync_image_0000000791_image_03.png")
        prediction[prediction < 0] = 0
        # visualize the prediction
        store_depth(prediction, "midas_prediction")
        # 5. Flatten the prediction
        prediction_flatten = prediction.flatten()
        # 6. Only choose the points where we exactly know the depth/disparity
        points = np.where(target_disparity != 0)[0]
        # 7. Calculate and align the prediction
        s,t = self.small_alignment(d=prediction_flatten, d_star=target_disparity, points=points)
        aligned_prediction = s*prediction + t
        aligned_prediction[aligned_prediction < self.disparity_cap] = self.disparity_cap
        # 8. Transform the disparity predition into metric depth
        aligned_prediction_inverted = 1.0 / aligned_prediction
        print(aligned_prediction_inverted.max())
        print(aligned_prediction_inverted.min())
        # visualize the metric depth and the aligned disparity
        store_depth(aligned_prediction, "midas_aligned_prediction")
        store_depth(aligned_prediction_inverted, "midas_metric_depth")
        self.first_aligning_run = False
        return s,t
    
    def small_alignment(self, d, d_star, points):
        """
        Calulate the shift and translation from the alignment
        This follows the caluclation part in the MiDaS paper

        Args:
            - d: disparity prediction
            - d_star: ground truth disparity
            - points: the points to align (where we definitely know the metric depth)
        """
        sum_1 = 0
        sum_2 = 0
        for i in points:
            di = np.array([
                [d[i]],
                [1]
            ])
            di_star = np.array([d_star[i]])
            sum_1 += np.matmul(di, di.transpose())
            sum_2 += np.matmul(di,di_star)
            h_opt = np.matmul(np.linalg.pinv(sum_1),sum_2)
        return h_opt






class MonoDepth2(DepthEstimator):
    def __init__(self, device, model_name, pred_metric_depth="True"):
        self.device = device
        self.pred_metric_depth = pred_metric_depth
        self.encoder, self.decoder, self.net_w, self.net_h = load_mono2_model(model_name, device)

    def _forward(self, img_rgb):
        img_rgb = img_rgb.to(self.device)
        features = self.encoder(img_rgb)
        outputs = self.decoder(features)
        return outputs
    
    def predict(self, img_path):
        with torch.no_grad():
            img_rgb = load_pil(img_path)
            original_width, original_height = img_rgb.size
            img_rgb = img_rgb.resize((self.net_w, self.net_h), Resampling.LANCZOS)
            img_rgb = transforms.ToTensor()(img_rgb).unsqueeze(0)

            outputs = self._forward(img_rgb)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
            if self.pred_metric_depth:
                depth = STEREO_SCALE_FACTOR * depth.cpu().numpy().squeeze()
        return depth


class DPT(DepthEstimator):
    def __init__(self, device, model_type, optimize=True, to_metric=True):
        self.first_aligning_run = True
        self.disparity_cap = 1.0 / 80
        self.optimize = optimize
        self.device = device
        model_path = dpt_default_models[model_type]
        model_path = "Depth_Estimation/DPT/" + model_path
        self.model_type = model_type
        self.model, self.transform, self.net_w, self.net_h = load_dpt_model(device, model_type, model_path, optimize)
        if to_metric:
            self.first_aligning_run = True
            self.scale, self.translation = self.compute_scale_and_shift()
    
    def _forward(self, sample, target_size):
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size,
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if self.model_type == "dpt_hybrid_kitti":
            prediction *= 256

        if self.model_type == "dpt_hybrid_nyu":
            prediction *= 1000.0

        return prediction
    
    def predict(self, img_path):
        img_rgb = read_image(img_path)
        img_input = self.transform({"image": img_rgb})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self._forward(sample, img_rgb.shape[:2])

            if not self.first_aligning_run:
                    prediction[prediction < 0] = 0
                    prediction_aligned = self.scale * prediction + self.translation
                    prediction_aligned[prediction_aligned < self.disparity_cap] = self.disparity_cap
                    prediction = prediction_aligned
            return prediction

    def compute_scale_and_shift(self):
        """
        Computes and sets the scale and shift for the aligned prediction
        -> is needed when performing metric depth instead of relative depth
        """
        # 1. Load the ground truth data from the kitti dataset
        depth_map, mask = depth_read("depth_selection/val_selection_cropped/groundtruth_depth/2011_10_03_drive_0047_sync_groundtruth_depth_0000000791_image_03.png")
        # 2. Transform the absolute grouth truth depth into disparity
        target_disparity = np.zeros_like(depth_map)
        target_disparity[mask == 1] = 1.0 / depth_map[mask == 1]
        # 3. Flatten the disparity for alignment
        target_disparity_flatten = target_disparity.flatten()
        # 4. Get the not yet aligned prediction of the model
        prediction = self.predict("depth_selection/val_selection_cropped/image/2011_10_03_drive_0047_sync_image_0000000791_image_03.png")
        prediction[prediction < 0] = 0
        # 5. Flatten the prediction
        prediction_flatten = prediction.flatten()
        # 6. Only choose the points where we exactly know the depth/disparity
        points = np.where(target_disparity_flatten != 0)[0]
        # 7. Calculate and align the prediction
        s,t = self.small_alignment(d=prediction_flatten, d_star=target_disparity_flatten, points=points)
        aligned_prediction = s*prediction + t
        aligned_prediction[aligned_prediction < self.disparity_cap] = self.disparity_cap
        # 8. Transform the disparity predition into metric depth
        aligned_prediction_inverted = 1.0 / aligned_prediction
        print(aligned_prediction_inverted.max())
        print(aligned_prediction_inverted.min())
        # visualize and store all the intermediate steps
        store_path = "assets/DPT/" + self.model_type
        store_depth(target_disparity, store_path + "_target_disparity")
        store_depth(depth_map, store_path + "_detph_map")
        store_depth(aligned_prediction, store_path + "_aligned_prediction")
        store_depth(aligned_prediction_inverted, store_path + "_metric_depth")
        store_depth(prediction, store_path +  "_prediction")
        # store_depth(target_disparity, store_path + "_target_disparity", format="pgf")
        # store_depth(depth_map, store_path + "_detph_map", format="pgf")
        # store_depth(aligned_prediction, store_path + "_aligned_prediction", format="pgf")
        # store_depth(aligned_prediction_inverted, store_path + "_metric_depth", format="pgf")
        # store_depth(prediction, store_path +  "_prediction", format="pgf")
        self.first_aligning_run = False
        return s,t
    def small_alignment(self, d, d_star, points):
            """
            Calulate the shift and translation from the alignment
            This follows the caluclation part in the MiDaS paper

            Args:
                - d: disparity prediction
                - d_star: ground truth disparity
                - points: the points to align (where we definitely know the metric depth)
            """
            sum_1 = 0
            sum_2 = 0
            for i in points:
                di = np.array([
                    [d[i]],
                    [1]
                ])
                di_star = np.array([d_star[i]])
                sum_1 += np.matmul(di, di.transpose())
                sum_2 += np.matmul(di,di_star)
                h_opt = np.matmul(np.linalg.pinv(sum_1),sum_2)
            return h_opt
class ZoeDepth(DepthEstimator):
    def __init__(self, device, model_type):
        if not (model_type in zoe_model_types):
            raise TypeError("Model type is not supported by ZoeDepth: use one of [ZoeD_N, ZoeD_K, ZoeD_NK]")
        
        repo = "isl-org/ZoeDepth"
        # Zoe_N
        self.model = torch.hub.load(repo, model_type, pretrained=True)
        self.model = self.model.to(device)

    def _forward(self, img_rgb, output_type="numpy"):
        if output_type == "numpy":
            prediction = self.model.infer_pil(img_rgb)
        else:
            prediction = self.model.infer_pil(img_rgb, output_type)
        
        return prediction
    
    def predict(self, img_path, output_type="numpy"):
        img_rgb = load_pil(img_path)
        return self._forward(img_rgb, output_type)
    