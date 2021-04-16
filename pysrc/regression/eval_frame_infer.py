import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
import yaml
import os
import csv

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms


import sys
sys.path.append('../')
from common import bnn_network

class FrameInferEval:
    def __init__(self,CFG):
        print("Eval Frame Infer")

        self.frame_infer_log_top_path = CFG["frame_infer_log_top_path"]
        self.frame_infer_log_file_name = CFG["frame_infer_log_file_name"]

        self.dataset_data_top_path = CFG["dataset_data_top_path"]
        self.dataset_data_file_name = CFG["dataset_data_file_name"]

        self.loop_period = CFG["loop_period"]

        self.do_eval()

    def do_eval(self):
        log_path = os.path.join(self.frame_infer_log_top_path, self.frame_infer_log_file_name)
        dataset_path = os.path.join(self.dataset_data_top_path, self.dataset_data_file_name)

        log_list = []
        with open(log_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                log_list.append(row)

        dataset_list = []
        with open(dataset_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                dataset_list.append(row)

        loop_bar = zip(log_list, dataset_list)
        
        for row_log, row_dataset in loop_bar:
            log_pic = cv2.imread(row_log[5])
            cv2.imshow('image_log',log_pic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




if __name__ == '__main__':

    parser = argparse.ArgumentParser("./eval_frame_infer.py")

    parser.add_argument(
        '--eval_frame_infer_config', '-efic',
        type=str,
        required=False,
        default='/home/ros_catkin_ws/src/bnn_attitude_predictor_with_image/config/eval_frame_infer_config.yaml',
        help='Eval frame infer config yaml file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.eval_frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.eval_frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.eval_frame_infer_config)
        quit()

    frame_infer_eval = FrameInferEval(CFG)