from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

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

def main():
    #Hyperparameters
    method_name = "regression"
    list_train_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/train"]
    list_val_rootpath = ["../../../dataset_image_to_gravity/AirSim/1cam/val"]

if __name__ == '__main__':
    main()