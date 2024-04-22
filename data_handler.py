import os
from enum import Enum
import cv2


class InputType(Enum):
    """
    CLASS that handles the input type used by the InputHandler
    """
    IMAGES = 1
    VIDEO = 2
    CAMERA = 3
    DIR = 4


class InputHandler:
    """
    CLASS serving as interface to the input/output files or directory
    """
    def __init__(self,
                 input_path:str,
                 output_path_depth: str,
                 output_path_seg: str):
        
        self.input_path = input_path
        self.output_path_depth = output_path_depth
        self.output_path_seg = output_path_seg
        self.input_images = []
        self.cap = None
        self.input_type = None
        self.image_extensions = (".jpg", ".png", ".jpeg")
        self.video_extensions = (".MOV", ".mp4", ".avi")

        # check if a valid input path was swet
        self.check_input_type()

        # create depth estimator output path
        if not os.path.exists(self.get_output_path_depth()):
            os.mkdir(self.get_output_path_depth())

    def check_input_type(self):
        """
        Checks the input path is a valid input type
        """
        # determine if input is a folder, video file, or camera
        if os.path.isdir(self.input_path):
            # input is a folder of images
            self.input_type = InputType.DIR
            for filename in os.listdir(self.input_path):
                if filename.endswith(self.image_extensions):
                    self.input_images.append(os.path.join(self.input_path, filename))

        elif self.input_path.isdigit():
            # input is a USB camera path
            self.input_type = InputType.CAMERA
            self.cap = cv2.VideoCapture(int(self.input_path)) 

        elif self.input_path.endswith(self.image_extensions):
            # input is an image
            self.input_type = InputType.IMAGES

        elif self.input_path.endswith(self.video_extensions):
            # input is a Video
            self.input_type = InputType.VIDEO
            self.cap = cv2.VideoCapture(self.input_path)
        else:
            raise TypeError(f"Wrong input type -> {os.path.splitext(self.input_path)[-1]} is not supported")

    def load_image(self, image_path):
        """
        Loads an image from path using OpenCV

        Args:
            - image_path: input path
        """
        return cv2.imread(image_path)

    def save_image(self, image, image_path): 
        """
        Saves an image to a given path

        Args:
            - image: numpy image
            - image_path: output_path
        """       
        # get the path without the file extension
        base_name = os.path.splitext(image_path)[0]  
        new_file_path = base_name + ".png"
        cv2.imwrite(os.path.join(self.output_path_seg, os.path.basename(new_file_path)), image)
    

    def get_images(self) -> list:
        """
        Getter function for returning the image paths
        """
        return self.input_images

    def get_input_type(self):
        """
        Getter function for returning the input type
        """
        return self.input_type

    def get_cap(self):
        """
        Getter function for getting the cap
        """
        return self.cap

    def get_output_path_seg(self):
        """
        Getter function for getting the segmentator output path
        """
        return self.output_path_seg
    
    def get_output_path_depth(self):
        """
        Getter function for getting the depth estimator output path
        """
        return self.output_path_depth
