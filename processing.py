from data_handler import InputHandler, InputType
import io_utils
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import cv2
import csv
from scipy.signal import convolve2d

class PerformanceTimer:
    """
    CLASS for tracking the executation times, calculate important statistical features and visualization of results
    """
    def __init__(self):
        self.infer_times = np.array([])
        self.stats = {}
        self.first_calibration_run = True
        self.division_factor = 1e6
    
    def get_statistics(self):
        """
        Calculation of statistical features of the acquired dataset of execution times
        """
        if self.infer_times.shape[0]:
            self.stats = {
                "max": np.amax(self.infer_times, axis=0),
            "mean": np.mean(self.infer_times, axis=0),
            "std": np.std(self.infer_times, axis=0)}
        else:
            raise Exception("The infer_times np.array does not contain any values")
        return self.stats
    
    def start_session(self):
        """Start a tracking session in nanoseconds 
        """
        self.current_tstamp = time.time_ns()
        return self.current_tstamp

    def end_session(self):
        """
        Ends a tracking session and computes the previously estimated nanoseconds into milliseconds
        Additionaly stores the duration of start and end tracking time
        """
        try:
            if self.first_calibration_run:
                print("Calibrate PerformanceTimer--")
                process_time = (time.time_ns() - self.current_tstamp) / self.division_factor
                print(process_time)
                if process_time > 1000:
                    self.division_factor = 1e9
                    process_time = process_time / 1e3
                self.first_calibration_run = False
            else:
                process_time = (time.time_ns() - self.current_tstamp) / self.division_factor
            
            if self.division_factor == 1e6:
                print("Exec. time:", process_time, "ms")
            else:
                print("Exec. time:", process_time, "s")
            self.infer_times = np.append(self.infer_times,process_time)
        except Exception as e:
            print(e)
            return

    def visualize(self, path, flag, store=False):
        """
        Visualization of the acquired duration times 

        Args:
            - path: output path
            - flag: for adding a description with model was used
            - store (default=False): additionally stores the plot if set to TRUE
        """
        self.get_statistics()
        x = np.arange(0,self.infer_times.shape[0],1)
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.infer_times, label="time", marker="x")
        plt.axhline(self.stats["mean"], color='r', linestyle='--', label='Mean')
        plt.axhline(self.stats["mean"] + self.stats["std"], color='g', linestyle='--', label='Mean + Std Dev')
        plt.axhline(self.stats["max"], color='b', linestyle='--', label='Max')
        plt.axhline(self.stats["mean"] - self.stats["std"], color='g', linestyle='--', label='Mean - Std Dev')
        plt.legend(loc="upper right")
        plt.title(flag)
        plt.xlabel('Index')
        if self.division_factor == 1e6:
            plt.ylabel('Inference time in (ms)')
        else:
            plt.ylabel('Inference time in (s)')
        plt.grid(True)
        if store:
            if not os.path.exists(os.path.join(path,"statistics")):
                os.mkdir(os.path.join(path,"statistics"))
            plt.savefig(os.path.join(path,"statistics", str(flag) + ".png"))
        else:
            plt.show()

class Processor:
    """
    CLASS as interface between the input data, the segmentation model and the depth estimator
    """
    def __init__(self, ih:InputHandler, segmentator=None, depth_estimator=None, estimator_type=None):
        """
        Args:
            - ih: InputHandler (class) which serves as interface to images or directories
            - segmentator (default=None): if not set -> no segmentation is done, otherwise provide Segmentator (class)
            - depth_estimator (default=None): if not set -> no depth estimation is done, otherwise provide a
              child class of base class DepthEstimator (class)
            - estimator_type (default=None): flag for which estimator is used
        """
        self.ih = ih
        self.ih_input_type = self.ih.get_input_type()
        print("Input type: ",ih.get_input_type())
        # flags for setting if segmentator or depth estimator should be used
        self.use_seg = False
        self.use_depth = False
        if segmentator:
            print("Segmentation activated ...")
            self.segmentator = segmentator
            self.p_timer_seg = PerformanceTimer()
            self.use_seg = True
        
        if depth_estimator:
            print("Depth Estimatior activated ...")
            self.depth_estimator = depth_estimator
            self.p_timer_depth = PerformanceTimer()
            self.estimator_type = estimator_type
            self.use_depth = True
        
        if not (self.use_depth or self.use_seg):
            raise ValueError("Either a Segmentator or a Depth Estimator has to be set")

        self.box_filter = np.ones((5, 5)) / 25  # Normalizing the kernel for averaging
        self.depth_masks = {}
        self.label_indexes = {}

    def process(self):
        """
        process the input data accordingly depending on the input type of the path
        """
        if self.ih_input_type == InputType.IMAGES:
            self.process_image(self.ih.input_path)
        elif self.ih_input_type == InputType.DIR:
            self.process_dir()
        elif self.ih_input_type == InputType.CAMERA:
            self.process_camera()
        elif self.ih_input_type == InputType.VIDEO:
            self.process_video()
        else:
            raise TypeError("Wrong input type: Make sure you have the provided input types (IMAGES,DIR,CAMERA,VIDEO)")

    def process_image(self, img_path):
        if self.use_seg:
            image = cv2.imread(img_path)   
            height, width, _ = image.shape
            self.p_timer_seg.start_session()
            mask = self.segmentator.inference(image)
            self.p_timer_seg.end_session()
            mask_rgb = self.segmentator.mask_to_rgb(mask)
            mask_rgb = cv2.resize(mask_rgb, (width, height))
            # mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
            # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ih.save_image(mask_rgb, img_path)
        if self.use_depth:
            self.p_timer_depth.start_session()
            depth = self.depth_estimator.predict(img_path)
            self.p_timer_depth.end_session()
            base_name = os.path.splitext(img_path)[0]
            output_path = os.path.join(self.ih.get_output_path_depth(), os.path.basename(base_name))
            io_utils.store_depth(depth=depth, path=output_path, flag=self.estimator_type)

        
        if self.use_depth and self.use_seg:
            self.segmentation_masks = self.segmentator.get_segmentation_masks()
            (depth_masks,mean_depths) = self.mean_depth_per_mask(depth)
            self.get_label_location(height, width)
            self.annotate_labels(image.copy())
    
    def process_dir(self):
        """
        Processes all the images in a given directory
        """
        #print("Process dir")
        max_i = len(self.ih.get_images())
        for i, image_path in enumerate(self.ih.get_images()):
            print("Process " + str(i+1) + "/" + str(max_i) + " --> " + image_path)
            self.process_image(image_path)

    def process_camera(self):
        print("Process camera")
        cap = self.ih.get_cap()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_path = self.ih.get_output_path()
        depth_out_path = os.path.join(output_path,self.estimator_type + "_camera.avi")
        seg_out_path = os.path.join(output_path,"Seg_camera.avi")
        depth_out = cv2.VideoWriter(depth_out_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
        seg_out = cv2.VideoWriter(seg_out_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mask_bgr = self.process_single_img(frame)
            depth_map = self.depth_estimator.get_depth_map()

            # Convert RGB to BGR (OpenCV uses BGR by default)
            cv2_depth_map = cv2.cvtColor(np.array(depth_map), cv2.COLOR_RGB2BGR)

            seg_out.write(mask_bgr)
            depth_out.write(cv2_depth_map)

            if cv2.waitKey(0.01) & 0xFF == ord('q'):
                break


        cap.release()
        depth_out.release()
        seg_out.release()

    def process_video(self):
        print("Process video")
        cap = self.ih.get_cap()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_path = self.ih.get_output_path()
        depth_out_path = os.path.join(output_path,self.estimator_type + ".avi")
        seg_out_path = os.path.join(output_path,"seg.avi")
        depth_out = cv2.VideoWriter(depth_out_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
        seg_out = cv2.VideoWriter(seg_out_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mask_bgr = self.process_single_img(frame)
            depth_map = self.depth_estimator.get_depth_map()

            # Convert RGB to BGR (OpenCV uses BGR by default)
            cv2_depth_map = cv2.cvtColor(np.array(depth_map), cv2.COLOR_RGB2BGR)

            seg_out.write(mask_bgr)
            depth_out.write(cv2_depth_map)

        cap.release()
        depth_out.release()
        seg_out.release()

    def return_timer_stats(self):
        """
        Storing (Plotting) of the statistical features of the acquired execution times
        """
        stats = []
        if self.use_seg:
            self.p_timer_seg.visualize(path=self.ih.get_output_path_seg(), flag="Segmentator", store=True)
            stats.append(self.p_timer_seg.get_statistics())
        if self.use_depth:
            self.p_timer_depth.visualize(path = self.ih.get_output_path_seg(), flag=self.estimator_type, store=True)
            stats.append(self.p_timer_depth.get_statistics())
        return stats

    def mean_depth_per_mask(self, depth):
        """
        Calculates the mean metric depth the segmented masks

        Args:
            - depth: numpy depth map
        """
        file_path = "mean_depth_per_object.csv"
        mean_depths = {}
        f = open(file_path, "w", newline="")
        writer = csv.writer(f, delimiter=" ")
        for key, value in self.segmentation_masks.items():
            resized_mask = cv2.resize(value, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.depth_masks[key] = resized_mask*depth
            mean_depths[key] = (np.sum(self.depth_masks[key])/ np.sum(resized_mask))
            writer.writerow("hello")
        f.close()
        return (self.depth_masks, mean_depths)
    
    def get_label_location(self, height, width):
        label_indexes = {}
        for key, value in self.segmentation_masks.items():
            resized_mask = cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST)
            convolved_matrix = convolve2d(resized_mask, self.box_filter, mode='same', boundary='fill', fillvalue=0)
            self.label_indexes[key] = np.unravel_index(np.argmax(convolved_matrix), convolved_matrix.shape)
        return self.label_indexes
    
    def annotate_labels(self,img):
        for (class_id_ref , (idx_x, idx_y)) in self.label_indexes.items():
            cv2.putText(img, str(self.segmentator.class_labels[class_id_ref]), (idx_y,idx_x), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,255), 1)
        # for (class_id, class_name), (class_id_ref , (idx_x, idx_y)) in zip(self.segmentator.class_labels.items(), self.label_indexes.items()):
        #     print(str(class_id) + ": " + class_name)
        #     print(str(class_id_ref) + ": " + "(" + str(idx_x) + ", " + str(idx_y) + ")")
        cv2.imwrite("example.png", img)

