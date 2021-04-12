from tqdm import tqdm
import matplotlib.pyplot as plt
import time
#import datatime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self,
        method_name,
        train_dataset,
        valid_dataset,
        net,
        criterion,
        optimizer_name,
        lr_cnn,
        lr_fc,
        batch_size,
        num_epochs):

        self.setRandomCondition()
        
        #gpu:0 -> Using GPU number 0
        #If it have multi GPU, be able to assign other GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        print("Training Device: ", self.device)

        self.dataloaders_dict = self.getDataloader(train_dataset, valid_dataset, batch_size)
        self.net = self.getSetNetwork(net)
        self.criterion = criterion
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter = self.getStrHyperparameter(method_name, train_dataset, optimizer_name, lr_cnn, lr_fc, batch_size)
        
    def setRandomCondition(self, keep_reproducibility=False): #Random Training Environment

        #Refer https://nuka137.hatenablog.com/entry/2020/09/01/080038

        if keep_reproducibility:
            torch.manual_seed(19981020)
            np.random.seed(19981020)
            random.seed(19981020)
            torch.backends.cudnn.deterministic = True #https://qiita.com/chat-flip/items/c2e983b7f30ef10b91f6
            torch.backends.cudnn.benchmark = False


    def getDataloader(self, train_dataset, valid_dataset, batch_size):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle = False
        )

        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getSetNetwork(self, net):
        print(net)

        net = net.to(self.device) #Send to GPU
        return net

    def getOptimizer(self, optimizer_name, lr_cnn, lr_fc):
        list_cnn_param_value, list_fc_param_value = self.net.getParamValueList()

        #Set Optimizer
        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value,  "lr": lr_fc },
            ], momentum=0.9)
        elif optimizer_name == "Adam":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value,  "lr": lr_fc },
            ])

        print(optimizer)
        return optimizer

    def getStrHyperparameter(self, method_name, dataset, optimizer_name, lr_cnn, lr_fc, batch_size):
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "train" \
            + str(len(self.dataloaders_dict["valid"].dataset)) + "valid" \
            + str(dataset.transform.resize) + "resize" \
            + str(dataset.transform.mean[0]) + "mean" \
            + str(dataset.transform.std[0]) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter
    
    def train(self):
        print("Train Debug")


