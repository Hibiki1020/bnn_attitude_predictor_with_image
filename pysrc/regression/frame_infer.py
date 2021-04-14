import cv2
from PIL import ImageMsg
import math
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

from common import bnn_network

class BnnAttitudeEstimationWithImageFrame(self, CFG):
    def __init__(self, CFG):
        print("BNNAttitudeEstimationWithImageFrame")
        
        self.CFG = CFG
        #contain yaml data to variance
        self.method_name = CFG["method_name"]

        self.dataset_frame_path = CFG["dataset_frame_path"]
        self.csv_name = CFG["csv_name"]
        self.weights_top_path = CFG["weights_top_path"]
        self.weights_file_name = CFG["weights_file_name"]
        
        self.log_file_path = CFG["log_file_path"]
        self.log_file_name = CFG["log_file_name"]

        self.frame_id = CFG["frame_id"]

        self.resize = CFG["resize"]
        self.mean_element = CFG["mean_element"]
        self.std_element = CFG["std_element"]
        self.num_mcsampling = CFG["num_mcsampling"]
        self.dropout_rate = CFG["dropout_rate"]

        #saving parameter in csv file
        self.v_vector = []
        self.accel_msg = []
        self.epistemic = []

        #open_cv
        self.bridge = CvBridge()
        self.color_img_cv = np.empty(0)

        #BNN
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(resize, mean_element, std_element)
        self.net = self.getNetwork(resize, weights_path)
        self.enable_dropout()




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("./frame_infer.py")

    parser.add_argument(
        '--frame_infer_config', '-fic',
        type=str,
        required=False,
        defalut='/home/ros_catkin_ws/src/bnn_attitude_predictor_with_image/config/frame_infer_config.yaml',
        help='Frame infer config yaml file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.frame_infer_config)
        quit()

    bnn_attitude_predictor_with_image_frame = BnnAttitudeEstimationWithImageFrame(CFG)