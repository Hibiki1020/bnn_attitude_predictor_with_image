import cv2
from PIL import ImageMsg
import math
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

from common import bnn_network


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

    #contain yaml data to variance
    method_name = CFG["method_name"]
    
    dataset_frame_path = CFG["dataset_frame_path"]
    csv_name = CFG["csv_name"]

    weights_top_path = CFG["weights_top_path"]
    weights_file_name = CFG["weights_file_name"]

    log_file_path = CFG["log_file_path"]
    log_file_name = CFG["log_file_name"]