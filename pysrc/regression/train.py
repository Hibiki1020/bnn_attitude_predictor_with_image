from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import bnn_network


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument{
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='../../config/train_config.yaml'
        help='Train hyperparameter config file'
    }
    parser.add_argument{
        '--log_place', '-l',
        type=str,
        required=False,
        default='../../weights/test1'
        help='Place to put trained file'
    }

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    #Create weights folder
    try:
        if FLAGS.log_place == "":
            
            if os.path.isdir(FLAGS.log_place):
                if os.listdir(FLAGS.log_place):
                    answer = raw_input("Log Directory is not empty. Do you want to proceed? [y/n]  ")
                    if answer == 'n':
                        quit()
                    else:
                        shutil.rmtree(FLAGS.log_place)
            os.mkdirs(FLAGS.log_place)
        else:
            print("Not creating new log file.")
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()
    
    try:
        print("Copy files to %s for further reference." % FLAGS.log)
        copyfile(FLAGS.train_cfg, FLAGS.log_place + "/train_config.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting....")
        quit()

    #Start inference
    trainer = Trainer(CFG, FLAGS.train_cfg, FLAGS.log_place)
    trainer.train()

