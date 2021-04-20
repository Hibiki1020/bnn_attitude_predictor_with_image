#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError

import cv2
import PIL.Image as Image
import math
import numpy as np
import argparse
import yaml
import os
import time

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import sys

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from common import bnn_network

class SaveROSMsg:

    def __init__(self):
        self.frame_id = rospy.get_param('~frame_id', '/base_link')
        
        self.onecam_checker = rospy.get_param('~1cam_checker',"True")
        #If this parameter is false, mode is changed to 4cam
        
        self.front_cam_topic = rospy.get_param('~front_cam_topic', '/camera_f/decompressed_image')
        self.left_cam_topic = rospy.get_param('~left_cam_topic', '/camera_l/decompressed_image')
        self.right_cam_topic = rospy.get_param('~right_cam_topic','/camera_r/decompressed_image')
        self.back_cam_topic = rospy.get_param('~back_cam_topic', '/camera_b/decompressed_image')

        self.velodyne_topic = rospy.get_param('~velodyne_topic', '/velodyne_packets')

        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data')

        self.wait_sec = float(rospy.get_param('~wait_sec', '3'))

        self.catch_imu_checker = False
        self.catch_img_checker = False

        if(self.onecam_checker==True):
            #OpenCV
            self.bridge = CvBridge()
            self.color_img_cv = np.empty(0)
            self.sub_image = rospy.Subscriber(self.front_cam_topic, ImageMsg, self.callbackColorImage, queue_size=1)
        else:
            print("sss")
            quit()
        
        self.imu_data = Imu()
        self.sub_imu_msg = rospy.Subscriber(self.imu_topic, Imu, self.callbackImuMsg, queue_size=1)

    def callbackImuMsg(self, msg):
        self.imu_data = msg
        #print("catch imu data")
        self.catch_imu_checker = True
            
    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.catch_img_checker = True
            print("Got Image msg")

            self.save_data()
            
            time.sleep(self.wait_sec) #wait X sec

            self.catch_img_checker = False
            self.catch_imu_checker = False
        except CvBridgeError as e:
            print(e)

    def save_data(self):
        print("save data")

def main():
    rospy.init_node('save_rosmsg', anonymous=True)

    save_rosmsg = SaveROSMsg()
    rospy.spin()

if __name__ == '__main__':
    main()