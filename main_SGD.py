'''
Local SGD with an increasing sample size sequence
'''

from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import json
import os
import sys
import time

import random
# hot fix
import random as rand

from datetime import datetime
import copy

from threading import Thread, Lock
import threading

import logging

#
from load_optim import load_optim
from evaluate import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
#
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

# bigger models
import AlexNet
import LeNet
import misc

import matplotlib.pyplot as plt
import pandas as pd

# plot the figures
import numpy as np

##### New configuration #####
import memory_block
import serialization as serial
import DP_config as config
import utils
import file_utils

# https://discuss.pytorch.org/t/segmentation-fault-problem/82917/3
# require torch>=1.5.0

# --------------------------------- logging config -------------------------------- #
# create logger with 'framework'
logger = logging.getLogger('framework')
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()  # stdout
# ch = logging.FileHandler('framework.log')    # logging
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s [%(threadName)-10.10s] {%(pathname)s:%(lineno)d} [%(levelname)-5.5s]  %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)
# --------------------------------- logging config -------------------------------- #


# This is the customized file for reimplementing the SGD class (in Pytorch)

# --------------------------- Utils --------------------------- $
import uuid

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

# print(my_random_string(6)) # For example, D9E50C
# --------------------------- Utils --------------------------- $


criterion = nn.CrossEntropyLoss()

# Ref: https://github.com/pytorch/examples/blob/master/mnist/main.py
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

# Defining the network (LeNet-5)  
# class LeNet5(torch.nn.Module):          
     
#     def __init__(self):     
#         super(LeNet5, self).__init__()
#         # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
#         self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
#         # Max-pooling
#         self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
#         # Convolution
#         self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
#         # Max-pooling
#         self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
#         # Fully connected layer
#         self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
#         self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
#         self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
        
#     def forward(self, x):
#         # convolve, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.conv1(x))  
#         # max-pooling with 2x2 grid 
#         x = self.max_pool_1(x) 
#         # convolve, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.conv2(x))
#         # max-pooling with 2x2 grid
#         x = self.max_pool_2(x)
#         # first flatten 'max_pool_2_out' to contain 16*5*5 columns
#         # read through https://stackoverflow.com/a/42482819/7551231
#         x = x.view(-1, 16*5*5)
#         # FC-1, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc1(x))
#         # FC-2, then perform ReLU non-linearity
#         x = torch.nn.functional.relu(self.fc2(x))
#         # FC-3
#         x = self.fc3(x)
        
#         return x

# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
#         # Add dropout function.
#         self.dropout1 = nn.Dropout2d(0.2)
#         #~ Dropout

#         # self.fc1 = nn.Linear(4*4*50, 500)
#         # self.fc2 = nn.Linear(500, 10)

#         self.fc1 = nn.Linear(4*4*50, 128)
#         self.fc2 = nn.Linear(128, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)

#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))

#         # Dropout
#         x = self.dropout1(x)
#         #~Dropout

#         x = F.relu(self.fc2(x))

#         x = self.fc3(x)
#         return x
    
#     def name(self):
#         return "LeNet5"
# ------------------ my Lenet


# class Swish(nn.Module):
#   def forward(self, input):
#     return (input * torch.sigmoid(input))
  
#   def __repr__(self):
#     return self.__class__.__name__ + ' ()'

# class CNNModel(nn.Module):
#     def __init__ (self):
#         super(CNNModel,self).__init__()
        
#         self.cnn1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1,padding=1)
#         self.bn1=nn.BatchNorm2d(16)
#         self.swish1=Swish()
#         nn.init.xavier_normal(self.cnn1.weight)
#         self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
#         self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1)
#         self.bn2=nn.BatchNorm2d(32)
#         self.swish2=Swish()
#         nn.init.xavier_normal(self.cnn2.weight)
#         self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
#         self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
#         self.bn3=nn.BatchNorm2d(64)
#         self.swish3=Swish()
#         nn.init.xavier_normal(self.cnn3.weight)
#         self.maxpool3=nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(64*6*6,10)
        
#         self.softmax=nn.Softmax(dim=1)
        
        
#     def forward(self,x):
#         out=self.cnn1(x)
#         out=self.bn1(out)
#         out=self.swish1(out)
#         out=self.maxpool1(out)
#         out=self.cnn2(out)
#         out=self.bn2(out)
#         out=self.swish2(out)
#         out=self.maxpool2(out)
#         out=self.cnn3(out)
#         out=self.bn3(out)
#         out=self.swish3(out)
#         out=self.maxpool3(out)
#         out=out.view(out.size(0),-1)
#         out=self.fc1(out)
#         out=self.softmax(out)
        
#         return out

##### New conifugration #####
class evalObject():
    def __init__(self):
        self.test_accuracy = []
        self.gradient_norm = []

    def get_test_accuracy(self):
        return self.test_accuracy

    def get_gradient_norm(self):
        return self.gradient_norm



class itemData(object):
    def __init__(self, my_input, my_label):
        self.input = my_input
        self.label = my_label

    def get_input(self):
        return self.input

    def get_label(self):
        return self.label

##### New configuration #####
current_global_model = None


# ------------------------------ Model ---------------------------------- #
# arr_sum_gradient = [] # plot
array_evalObj = []
# adapt to queue
queue_gradient = [] # contain sum of gradients
# queue_gradient = memory_block.QueueGradient()
global_model = None
temp_global_model = None
stop_signal = False # stopping?

# New configuration
arr_duration = []
# ~configuration


list_sample_sequence = None # s_{i,c}?
list_stepsize_sequence = None # eta_i=?
list_increasing_iteration = None # plot

##### New configuration #####
neighborMap = {} # empty map
neighborEdgeMapWrite = {}
neighborEdgeMapRead = {}

# synchronized shared memory
synchronized_memory = None

# Epoch counter
epochCounterMap = {} # empty map
epoch_mutex_queue = Lock()

# counting the number of gradient sending
num_send_gradient_mutex = Lock()
num_send_gradient = 0

# stop flag
stop_signal = False


def update_num_send_gradient():
    global num_send_gradient
    global num_send_gradient_mutex

    num_send_gradient_mutex.acquire()
    try:
        num_send_gradient = num_send_gradient + 1
    finally:
        num_send_gradient_mutex.release()



def get_gradient_queue_by_name(queue_name=""):
    global queue_gradient
    for queue in queue_gradient:
        name = queue.get_queue_name()
        if name == queue_name:
            return queue

    return None


# Algorithm 1 Client_c – Local asynchronous SGD – Adaptive \tau
def is_synchronize_needed(client_id=0, d_asynchronous_round=0, current_round=0):
    global neighborMap
    global epochCounterMap

    neighbors = []
    # neighbors.append(client_id)

    # get list of neighbors by client_id
    temp_neighbors = neighborMap[client_id]
    for element in temp_neighbors:
        neighbors.append(element)

    d_round = 0.0
    for element in neighbors:
        # get neighbors of element
        element_neighbors = neighborMap[element] # always > 0
        element_round = int(float(epochCounterMap[element]) / float(len(element_neighbors)))
        # logger.debug("Client {}: element_round={}".format(element, element_round))
        d_round = max(d_round, current_round - element_round)

    # checking...
    if d_round > float(d_asynchronous_round):
        logger.info("Client {}: element_round={}, d_round={}".format(element, element_round, d_round))
        return True
    return False


# new synchronization point...
def is_synchronize_needed_method2(client_id=0, current_epoch=0, d_asynchronous_round=2):
    global synchronized_memory

    # filter the epoch values
    epoch_value = []
    for value in synchronized_memory:
        if value != misc.FLAG_FINISHED_EPOCH:
            epoch_value.append(value)

    assert(len(epoch_value) > 0)

    # wait for all nodes
    min_epoch = min(epoch_value)
    if abs(current_epoch - min_epoch) <= d_asynchronous_round:
        return False
    else:
        logger.info("Client {}: min_round={}, d_round={}".format(client_id, min_epoch, d_asynchronous_round))
        return True


def get_round_method(client_id=0, current_epoch=0):
    global synchronized_memory

    # filter the epoch values
    min_epoch = current_epoch
    epoch_value = []
    for value in synchronized_memory:
        if value != misc.FLAG_FINISHED_EPOCH:
            epoch_value.append(value)

    # assert(len(epoch_value) > 0)

    # wait for all nodes
    if len(epoch_value) > 0:
        min_epoch = min(epoch_value)
    # if abs(current_epoch - min_epoch) <= d_asynchronous_round:
    #     return False
    # else:
    #     logger.info("Client {}: min_round={}, d_round={}".format(client_id, min_epoch, d_asynchronous_round))
    #     return True

    return min_epoch


# ------------------------------ Model ---------------------------------- #

def train(args, model, device, train_loader, optimizer, epoch, client_id, internal_queue_gradient, client_data):
    global array_evalObj
    # global queue_gradient
    global list_sample_sequence
    global list_stepsize_sequence
    global queue_gradient
    global neighborMap
    global neighborEdgeMapWrite
    global neighborEdgeMapRead
    global epochCounterMap
    global epoch_mutex_queue

    # for synchronized purpose
    global synchronized_memory


    model.train()
    # DP_CLIPPING_NORM = 0.1
    DP_CLIPPING_NORM = 0.1
    
    # sample size at epoch
    current_epoch_sample = list_sample_sequence[epoch]
    # logger.info("Client {} inside train: current_epoch_sample={}".format(client_id, current_epoch_sample))

    need_repeat = True

    # 
    data_len = len(client_data)

    local_SGD_idx = 0
    train_loss = 0.0
    while (need_repeat == True):
        # local_SGD_idx = 0
        # local_train_loader = copy.deepcopy(train_loader)

        # for batch_idx, data in enumerate(local_train_loader, 0):
        for batch_idx in range(data_len):
            # randomize the data.
            r1 = random.randint(0, data_len-1)

            # get data from array
            copy_data = client_data[r1]

            # Set data
            data = copy_data.get_input()
            target = copy_data.get_label()

            # data, target = data # new way to access data
            # logger.info("Client {} : get data, batch_idx={}".format(client_id, batch_idx))
            data, target = data.to(device), target.to(device)
            # data, target = data, target
            # logger.info("Client {} : zero_grad, batch_idx={}".format(client_id, batch_idx))
            optimizer.zero_grad()
            # logger.info("Client {} : model(data), batch_idx={}".format(client_id, batch_idx))
            output = model(data)
            # logger.info("Client {} : F.cross_entropy(output, target), batch_idx={}".format(client_id, batch_idx))
            loss = F.cross_entropy(output, target, reduction='mean')
            train_loss = train_loss + loss.item()
            # logger.info("Client {} : backward, batch_idx={}".format(client_id, batch_idx))
            loss.backward()

            # clip gradients
            # ref: https://stackoverflow.com/questions/62466804/model-parameters-does-not-produce-an-iterable-of-tensors
            # PyTorch's clip_grad_norm, as the name suggests, operates on gradients. 
            # You have to calculate your loss from output, use loss.backward() 
            # and perform gradient clipping afterwards.
            # Also, you should use optimizer.step() after this operation.
            # You don't have parameter.grad calculated (it's value is None) 
            # and that's the reason of your error.

            # update internal model
            # logger.info("Client {} : optimizer.step(), batch_idx={}".format(client_id, batch_idx))
            optimizer.step()


            # check if we run enough iterations...
            local_SGD_idx = local_SGD_idx + 1

            ##################################################################
            if (local_SGD_idx % args.log_interval == 0 and local_SGD_idx > 0) or (local_SGD_idx == current_epoch_sample):
                # temp_sum_gradient_norm = optimizer.get_sum_gradient_norm()
                # addr_model = hex(id(model))
                # addr_optimizer = hex(id(optimizer))
                # addr_train_loader = hex(id(train_loader))
                # logger.debug("Client {} : batch_idx={}, addr_optimizer={}, train_loader={}".format(client_id,
                #                                                                     local_SGD_idx,
                #                                                                     addr_optimizer, 
                #                                                                     addr_train_loader))

                # logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, 
                #     batch_idx * len(data), 
                #     len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), 
                #     loss.item()))

                temp_sum_gradient_norm = optimizer.get_sum_gradient_norm()
                logger.info('Client {} : Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}\tNorm*LR:{:.6f}'.format(
                    client_id,
                    epoch,
                    args.epochs,
                    local_SGD_idx, 
                    current_epoch_sample,
                    train_loss/float(local_SGD_idx),
                    temp_sum_gradient_norm))


            ##### New configuration #####
            # Check for every iteration
            # We have new sum of gradient
            # We check the neighbor every iteration
            # automatically
            neighbor_name = neighborEdgeMapRead[client_id]
            logger.debug("Client {} has neighbors {}".format(client_id, neighbors))
            for each_neighbor_name in neighbor_name:
                # query queue_gradient from neighbors
                neighbor_queue_gradient = get_gradient_queue_by_name(queue_name=each_neighbor_name)

                while (neighbor_queue_gradient != None and neighbor_queue_gradient.is_empty() == False):
                    # we have new updated sum of gradient
                    gradInfo = neighbor_queue_gradient.dequeue()
                    if gradInfo is None:
                        continue

                    update_grad = gradInfo.get_gradient_sum()
                    # update_grad = update_grad.to("cpu") # cpu
                    update_grad = update_grad.to("cuda") # cuda
                    update_client_id = gradInfo.get_client_id()

                    ##### New configuration #####
                    # for safe: lock the epochCounterMap memory
                    # ref: https://stackoverflow.com/questions/3310049/proper-use-of-mutexes-in-python
                    epoch_mutex_queue.acquire()
                    # epochCounterMap[update_client_id] = epochCounterMap[update_client_id] + 1
                    try:
                        # print('Do some stuff')
                        epochCounterMap[update_client_id] = epochCounterMap[update_client_id] + 1
                    finally:
                        # mutex.release()
                        epoch_mutex_queue.release()

                    # check gradient norm
                    total_norm = update_grad.norm(2)
                    total_norm = total_norm ** (1. / 2.0)

                    # Update global model
                    # old_norm = global_model.norm(2)
                    # old_norm = old_norm ** (1. / 2.0)

                    # w = w  - lr*U
                    # convert model to vector form
                    temp_model_weight = serial.ravel_model_params(model, grads=False)
                    temp_model_weight.add_(1.0, update_grad)

                    # convert vector form to model

                    # model.add_(1.0, update_grad)
                    serial.unravel_model_params(model, temp_model_weight)

                    # new_norm = global_model.norm(2)
                    # new_norm = new_norm ** (1. / 2.0)

                    logger.info("Client {} reads gradient from client {}, grad_norm={:6f}, queue_name={}".format(client_id, 
                                                                                                update_client_id, 
                                                                                                total_norm,
                                                                                                each_neighbor_name))
                    # logger.info("Server updates gradient_norm from {:6f} to {:6f}".format(old_norm, new_norm))
                    # update_counter = update_counter + 1

                # else:
                #     time.sleep(0.01) # 10ms
            # LOOP NEIGHBORHOOD

            ##### New configuration #####
            # logger.info("Client {} : epoch {} : [snapshot] EpochMap is {}".format(client_id, epoch, epochCounterMap))

            # is_waiting = True
            # asynchronous_round = (2) # fixed value
            # num_trial = 0
            # while (is_waiting == True):
            #     bool_is_synchronize = is_synchronize_needed(client_id=client_id,
            #                                                 d_asynchronous_round=asynchronous_round,
            #                                                 current_round=epoch)
            #     if bool_is_synchronize == True:
            #         # wait for a while
            #         logger.warn("Client {} is waiting for its neighbors: asynchronous_round={}".format(client_id, asynchronous_round))
            #         time.sleep(1.0) # 5ms
            #         logger.info("Client {} : [snapshot] EpochMap is {}".format(client_id, epochCounterMap))
            #         num_trial = num_trial + 1
            #         if num_trial > 20:
            #             logger.error("Client {} increases the epochCounterMap: {}".format(client_id, epochCounterMap))

            #             # increase
            #             # this constraint is not  very strict
            #             # sometimes, we can relax this constraint to make the program run smoothly.
            #             list_neighnors = neighborMap[client_id]
            #             for element_neighbor in list_neighnors:
            #                 epoch_mutex_queue.acquire()
            #                 # epochCounterMap[update_client_id] = epochCounterMap[update_client_id] + 1
            #                 try:
            #                     # print('Do some stuff')
            #                     epochCounterMap[element_neighbor] = epochCounterMap[element_neighbor] + 1
            #                 finally:
            #                     # mutex.release()
            #                     epoch_mutex_queue.release()

            #             # break
            #             is_waiting = False
            #             break
            #     else:
            #         is_waiting = False
            #         break
                

            if local_SGD_idx == current_epoch_sample:
                logger.info("Client {} runs enough {} iterations at epoch={}".format(client_id, current_epoch_sample, epoch))

                # reset flag
                need_repeat = False

                # return...
                break

        # INNER LOPP

    # ~finish all SGD sequences

    ##### New configuration #####
    sum_gradient_norm = optimizer.get_sum_gradient_norm().item()
    logger.info("Client {} : Train Epoch {}, Norm*LR:{:.6f}".format(client_id, epoch, sum_gradient_norm))
    temp_grad_norm = copy.deepcopy(sum_gradient_norm)
    array_evalObj[client_id].get_gradient_norm().append(temp_grad_norm)

    # Add noise to the sum of gradients
    sum_gradient = optimizer.get_sum_gradient()
    if config.DP_EPSILON != float(config.MAGIC_EPSILON) and config.ENABLE_DP == misc.FLAG_ENABLE:
        noise_val = (config.DP_SIGMA * config.DP_CLIPPING_NORM)**2
        # fix bug: RuntimeError: normal_ expects std > 0.0, but found std=0
        if noise_val >= 0.0:
            noise = torch.FloatTensor(sum_gradient.shape).normal_(0, noise_val)
        else:
            noise = torch.zeros(sum_gradient.shape) # zero noise

    # add noise here.
    lr = optimizer.get_stepsize()
    # U = U + noise
    if config.DP_EPSILON != float(config.MAGIC_EPSILON) and config.ENABLE_DP == misc.FLAG_ENABLE:
        # noise = noise.to("cuda") # cuda?
        sum_gradient.add_(lr, noise)
        logger.info("Client {} : Train Epoch {} add noise, sigma={}, p={}, eps={}".format(client_id, epoch, 
                                                                        config.DP_SIGMA,
                                                                        config.DP_EXP_P,
                                                                        config.DP_EPSILON))
    # else:
    #     logger.info("Client {} : Train Epoch {} add noise, sigma={}, p={}, eps={}".format(client_id, epoch, 
    #                                                                     config.DP_SIGMA,
    #                                                                     config.DP_EXP_P,
    #                                                                     config.DP_EPSILON))
    #~noise...

    # device = torch.device("cuda")
    # noise = noise.to(device) # cuda?

    # add noise to model.
    temp_model_weight = serial.ravel_model_params(model, grads=False)
    # device = torch.device("cuda")
    # noise = noise.to(device) # cuda?
    # w = w + eta * noise
    if config.DP_EPSILON != float(config.MAGIC_EPSILON) and config.ENABLE_DP == misc.FLAG_ENABLE:
        device = torch.device("cuda")
        noise = noise.to(device) # cuda?

        temp_model_weight.add_(lr, noise)
        serial.unravel_model_params(model, temp_model_weight)

    # Update sum of gradients
    gradInfo = memory_block.GradientInformation(gradient_sum=sum_gradient, 
                                                local_round=epoch,
                                                client_id=client_id)

    ##### New configuration #####
    # Automatically get list of queue
    # write the sum of gradients to queue
    list_queue_name = neighborEdgeMapWrite[client_id]
    for queue_name in list_queue_name:
        updated_queue_gradient = get_gradient_queue_by_name(queue_name=queue_name)
        if updated_queue_gradient is None:
            continue

        updated_queue_gradient.enqueue(gradInfo)

        # counting the number of gradient sending
        update_num_send_gradient()
        # ~counting

        logger.info("Client {} writes gradient to queue_name={}".format(client_id, queue_name))

    # ref: https://bugs.python.org/issue35844
    # Uncommenting any of the `time.sleep`s makes it work consistently again.
    # "Indicate that no more data will be put on this queue by the
    # current process." --Documentation
    time.sleep(0.001 * 1) # 3ms
    # ~issue35844

    logger.debug("Client {} : Train Epoch {}, Set Gradient norm".format(client_id, epoch))
    optimizer.reset_sum_gradient()
    logger.debug("Client {} : Train Epoch {}, Reset Gradient norm".format(client_id, epoch))
    #### ~New configuration #####



def test(args, model, device, test_loader, epoch=0, total_epoch=0, client_id=0):
    global array_evalObj

    model.eval()

    # criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.cross_entropy(output, target, reduction='mean').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_acc = 100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     test_acc))

    # logger.info('{}/{} Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, total_epoch,
    #                                             correct, len(test_loader.dataset), test_acc))

    # arrayAccuracy.append(test_acc/100.)
    temp_accuracy = test_acc / 100.0
    # array_evalObj[client_id].get_test_accuracy().append(temp_accuracy)
    return temp_accuracy, test_loss # not in %


# a simple custom collate function, just to show the idea
# batch_size = 10240

def my_collate(batch):
    # global batch_size
    modified_batch = []
    # print("[debug] batch_size = {}".format(len(batch)))
    for item in batch:
        image, label = item
        # label_arr = hashLabel[dist.get_rank()]
        # if label in label_arr:
        modified_batch.append(item)

    return default_collate(modified_batch)


# only five clients
# each client has two subclass digits
hashLabel = {}
# hashLabel[0]=[0, 1, 2, 3, 4, 5, 6, 7, 8]
# hashLabel[1]=[1, 2, 3, 4, 5, 6, 7, 8, 9]
# hashLabel[2]=[2, 3, 4, 5, 6, 7, 8, 9, 0]
# hashLabel[3]=[3, 4, 5, 6, 7, 8, 9, 0, 1]
# hashLabel[4]=[4, 5, 6, 7, 8, 9, 0, 1, 2]

# 3 classes
# hashLabel[0]=[0, 1, 2, 3, 4]
# hashLabel[1]=[2, 3, 4, 5, 6]
# hashLabel[2]=[4, 5, 6, 7, 8]
# hashLabel[3]=[6, 7, 8, 9, 0]
# hashLabel[4]=[8, 9, 0, 1, 2]


# first 5 clients
hashLabel[0]=[0, 1, 2, 3, 4, 5, 6] # 2/3
hashLabel[1]=[2, 3, 4, 5, 6, 7, 8]
hashLabel[2]=[4, 5, 6, 7, 8, 9, 0]
hashLabel[3]=[6, 7, 8, 9, 0, 1, 2]
hashLabel[4]=[8, 9, 0, 1, 2, 3, 4]
# # next 5 clients
# hashLabel[5]=[0, 1, 2, 3, 4, 5, 6]
# hashLabel[6]=[2, 3, 4, 5, 6, 7, 8]
# hashLabel[7]=[4, 5, 6, 7, 8, 9, 0]
# hashLabel[8]=[6, 7, 8, 9, 0, 1, 2]
# hashLabel[9]=[8, 9, 0, 1, 2, 3, 4]

# ref: https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/12
def get_indices(dataset=None, client_id=0):
    global hashLabel

    indices =  []
    try:
        label_arr = hashLabel[client_id]
    except Exception as expt_label:
        logger.error("{}".format(str(expt_label)))
        os._exit(-1)

    # print(label_arr)
    logger.info("Client {} has the subdataset {}".format(client_id, label_arr))
    for i in range(len(dataset.targets)):
        # print('target={}'.format(dataset.targets[i]))
        if dataset.targets[i] in label_arr:
            indices.append(i)
    return indices

############################################################


# def central_receiver(thread_name="thread_receiver", args=None, internal_queue_gradient=None):
#     # global queue_gradient
#     global global_model
#     global current_global_model
#     global temp_global_model # server model
#     global array_evalObj
#     global stop_signal


#     # Loop for total_epoch
#     current_epoch = 0
#     total_epoch = args.epochs

#     num_client = args.num_clients

#     # Init the dataset
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

#     # load train dataset.
#     # Dataloader...
#     normalized_val = [0,0]
#     if args.dataset == 0: # mnist
#         dataset_loader = datasets.MNIST
#         logger.info("[server] We are using dataset MNIST...")

#         # hardcode
#         normalized_val[0] = 0.1307
#         normalized_val[1] = 0.3081

#     elif args.dataset == 1: # fashionMNIST
#         dataset_loader = datasets.FashionMNIST
#         logger.info("[server] We are using dataset FashionMNIST...")

#         normalized_val[0] = 0.5
#         normalized_val[1] = 0.5


#     # load test dataset.
#     test_loader = torch.utils.data.DataLoader(
#         dataset_loader('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((normalized_val[0],), (normalized_val[1],))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=False, **kwargs)

#     # init global model
#     # global_model

#     while (current_epoch < total_epoch):
#         update_counter = (0)

#         # switch to new global model
#         # when receiving enough models from client
#         # for simplicty, we set d=0
#         while(update_counter < num_client):
#             # check the queue
#             if internal_queue_gradient.is_empty() == False:
#                 # we have new updated sum of gradient
#                 gradInfo = internal_queue_gradient.dequeue()
#                 update_grad = gradInfo.get_gradient_sum()
#                 # update_grad = update_grad.to("cpu") # cpu
#                 update_grad = update_grad.to("cuda") # cuda
#                 update_client_id = gradInfo.get_client_id()

#                 # check gradient norm
#                 total_norm = update_grad.norm(2)
#                 total_norm = total_norm ** (1. / 2.0)

#                 # Update global model
#                 # old_norm = global_model.norm(2)
#                 # old_norm = old_norm ** (1. / 2.0)

#                 # w = w  - lr*U
#                 global_model.add_(1.0, update_grad)

#                 # new_norm = global_model.norm(2)
#                 # new_norm = new_norm ** (1. / 2.0)

#                 logger.info("Server updates gradient from client {}, grad_norm={:6f}".format(update_client_id, total_norm))
#                 # logger.info("Server updates gradient_norm from {:6f} to {:6f}".format(old_norm, new_norm))
#                 update_counter = update_counter + 1

#             else:
#                 time.sleep(0.01) # 10ms

#         # ----------- enough information ------------- #
#         current_epoch = current_epoch + 1
#         if current_epoch <= total_epoch:
#             # Update new global model
#             current_global_model.update_model(new_global_model=global_model)


#             temp_w = current_global_model.get_global_model()
#             addr_w = hex(id(temp_w))
#             logger.debug("[server] addr_w={}, epoch={}".format(addr_w, current_epoch))

#             # Notify new model
#             current_global_model.notify_new_model()

#             # Update server's model
#             serial.unravel_model_params(temp_global_model, global_model)
#             test_accuracy = test(args, 
#                                 temp_global_model, 
#                                 device, 
#                                 test_loader, 
#                                 current_epoch, 
#                                 args.epochs, 
#                                 0)
#             # array_evalObj[0].get_test_accuracy().append(test_accuracy)
#             logger.info("Server has test_accuracy={:.6f}, epoch={}".format(test_accuracy*100.0, current_epoch))

    
#     #### New configuration #####
#     logger.info("Stop ServThread with threadName={}".format(thread_name))
#     stop_signal = (True) # stopping=?



def main_function(thread_name="Thread_0", thread_id=1, args=None, client_id=0, internal_queue_gradient=None, client_data=[]):
    # global array_evalObj
    # global global_model
    global current_global_model
    global stop_signal
    global list_stepsize_sequence
    global epochCounterMap
    global neighborMap

    # New configuration
    global arr_duration

    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Running this code will reveal that the random number obtained each time is fixed. 
    # But if you do not add the function call to torch.manual_seed, 
    # the printed random number will be different each time.
    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    # ref: https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    # pin_memory: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    # The safest approach is using num_workers=0
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}
    # ~num_workers=0

    # load train dataset.
    # Dataloader...
    normalized_val = [0,0]
    if args.dataset == misc.MNIST_DATASET: # mnist
        dataset_loader = datasets.MNIST
        logger.info("[client] We are using dataset MNIST...")

        # hardcode
        normalized_val[0] = 0.1307
        normalized_val[1] = 0.3081

    elif args.dataset == misc.FASHION_MNIST_DATASET: # fashionMNIST
        dataset_loader = datasets.FashionMNIST
        logger.info("[client] We are using dataset FashionMNIST...")

        normalized_val[0] = 0.5
        normalized_val[1] = 0.5
        # normalized_val[0] = 0.1307
        # normalized_val[1] = 0.3081


    elif args.dataset == misc.KMNIST_DATASET: # EMNIST
        dataset_loader = datasets.KMNIST
        logger.info("[client] We are using dataset KMNIST...")

        normalized_val[0] = 0.5
        normalized_val[1] = 0.5
        # normalized_val[0] = 0.1307
        # normalized_val[1] = 0.3081

    elif args.dataset == misc.CIFAR_10:
        logger.info("[client] We are using dataset CIFAR10...")
        dataset_loader = datasets.CIFAR10


    # load train dataset.
    # train_loader = torch.utils.data.DataLoader(
    #     dataset_loader('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((normalized_val[0],), (normalized_val[1],))
    #                    ])),
    #     batch_size=args.batch_size,
    #     # collate_fn=my_collate, # use custom collate function here
    #     shuffle=True, 
    #     **kwargs)


    # load test dataset.
    if args.dataset != misc.CIFAR_10:
        test_loader = torch.utils.data.DataLoader(
            dataset_loader('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((normalized_val[0],), (normalized_val[1],))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_set = dataset_loader(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    #################################################################################


    # model = LeNet5().to(device) # letnet5

    if (args.dataset == misc.MNIST_DATASET) or (args.dataset == misc.FASHION_MNIST_DATASET) or (args.dataset == misc.KMNIST_DATASET):
        # model = LeNet5().to(device) # letnet5
        model = LeNet.myLeNet().to(device)
    elif args.dataset == misc.CIFAR_10:
        model = AlexNet.AlexNet().to(device)

    optimizer = load_optim(params=model.parameters(),
                           optim_method=args.optim_method,
                           eta0=args.eta0,
                           alpha=args.alpha,
                           milestones=args.milestones,
                           nesterov=args.nesterov,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           epoch_size=1000,
                           decay_steps=20000,
                           model=model,
                           is_customized=args.custom)

    batch_size = args.batch_size


    ##### New configuation #####
    temp_w = current_global_model.get_global_model()
    addr_w = hex(id(temp_w))
    logger.debug("[client] addr_w={}, epoch={}".format(addr_w, 0))
    # copy...
    serial.unravel_model_params(model, temp_w)
    my_model_id = current_global_model.get_uuid()
    # list_id.append(my_model_id)
    current_global_model.reset_value_at_block(block_index=client_id)

    ##### New configuration #####
    # client_data = []
    logger.info("# Client {} has client_data: len={}".format(client_id, 
                                                            len(client_data)))

    # start the training process...
    total_client_duration = 0
    for epoch in range(0, args.epochs):
        # reload train_loader
        # our goal: DataLoader will shuffle the new train_loader
        # This is a very cucumber data_loader source code
        # We cover both i.i.d and heterogeneous dataset
        # and cover many datsets, such as mnist, fashionmnist and cifar10
        # TODO: will seperate this part into a data_loader class later
        # if (config.ENABLE_BIASED_DATASET == misc.IID_DATASET) and len(client_data)==0: # i.i.d
        #     # lazy code
        #     # will refactor code

        #     if args.dataset != misc.CIFAR_10:
        #         train_loader = torch.utils.data.DataLoader(
        #             dataset_loader('../data', train=True, download=True,
        #                 transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((normalized_val[0],), (normalized_val[1],))
        #                 ])),
        #             batch_size=args.batch_size,
        #             # collate_fn=my_collate, # use custom collate function here
        #             shuffle=True, 
        #             **kwargs)
        #     else:
        #         transform_train = transforms.Compose([
        #             transforms.RandomCrop(32, padding=4),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #         ])

        #         train_set = dataset_loader(root='../data', train=True, download=True, transform=transform_train)
        #         train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        #     # # Create client_data
        #     # for batch_idx, data in enumerate(train_loader, 0):
        #     #     m_inputs, m_labels = data
        #     #     item = itemData(my_input=copy.deepcopy(m_inputs), my_label=copy.deepcopy(m_labels))
        #     #     client_data.append(item)

        #     # logger.info("Client {} has client_data: len={}".format(client_id, 
        #     #                                                         len(client_data)))


        # elif (config.ENABLE_BIASED_DATASET != misc.IID_DATASET) and len(client_data)==0: # heterogeneous
        #     if args.dataset != misc.CIFAR_10:
        #         train_set = dataset_loader(root='../data', train=True, 
        #                                                 download=True, 
        #                                                 transform=transforms.Compose([
        #                                                         transforms.ToTensor(),
        #                                                         transforms.Normalize((normalized_val[0],), (normalized_val[1],))
        #                                                 ]))
        #         idx = get_indices(train_set, client_id)
        #         # print(idx)
        #         # print(len(idx))
        #         logger.info("Client {} : idx_len={}".format(client_id, len(idx)))
        #         if len(idx) <= 0:
        #             logger.error("Client {} : invalid subdataset".format(client_id))
        #             os._exit(-1)

        #         train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, 
        #                                                 shuffle=False, 
        #                                                 drop_last=True,
        #                                                 sampler=SubsetRandomSampler(idx))
        #     else:
        #         transform_train = transforms.Compose([
        #             transforms.RandomCrop(32, padding=4),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #         ])

        #         train_set = dataset_loader(root='../data', train=True, download=True, transform=transform_train)
        #         # train_loader = data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, **kwargs)
        #         idx = get_indices(train_set, client_id)
        #         train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, 
        #                                                 shuffle=False, 
        #                                                 drop_last=True,
        #                                                 sampler=SubsetRandomSampler(idx))

        # # not equal dataset
        # ######################################################################################################
        # if len(client_data) <= 0:
            
        #     data_start_time = time.time() # start

        #     client_data_len = 0
        #     logger.info("Client {} start to calculate dataset...".format(client_id))
        #     # for batch_idx, data in enumerate(train_loader, 0):
        #     #     client_data_len = client_data_len + 1
        #     client_data_len = len(train_loader)

        #     #### Increase the client_data_len
        #     # client_data_len = int(client_data_len * args.num_clients)
        #     # client_data_len = int(client_data_len * 2.0)
        #     # client_data_len = int(client_data_len * 2.5)
        #     # client_data_len = int(client_data_len * 1.5)
        #     client_data_len = client_data_len // args.num_clients
        #     client_data_len = int(client_data_len * 1.5)


        #     if config.REQUIRE_NOT_EQUAL_DATASET == misc.FLAG_ENABLE: # heterogeneous
        #         fraction_len = 1.0 / (1.0 + 0.5 * client_id)
        #         client_data_len = int(fraction_len * client_data_len)
        #         logger.info("Client {} calculates dataset len={}".format(client_id, client_data_len))
        #     elif config.REQUIRE_SPLIT_DATASET == misc.FLAG_ENABLE: # split i.i.d
        #         fraction_len = 1.0 / float(args.num_clients)
        #         client_data_len = int(client_data_len * fraction_len)
        #         logger.info("Client {} calculates split_dataset len={}".format(client_id, client_data_len))

        #     # Create client_data
        #     # for batch_idx, data in enumerate(train_loader, 0):
        #     #     m_inputs, m_labels = data
        #     #     # using copy.deepcopy
        #     #     # thread-safe
        #     #     item = itemData(my_input=copy.deepcopy(m_inputs), my_label=copy.deepcopy(m_labels))
        #     #     if batch_idx <= client_data_len:
        #     #         client_data.append(item)
        #     #     else:
        #     #         break
        #     is_need_random = True
        #     rand_counter = 0
        #     # general while
        #     while (is_need_random == True):
        #         # ----------------- random local dataset ------------------- #
        #         for batch_idx, data in enumerate(train_loader, 0):
        #             m_inputs, m_labels = data
        #             # using copy.deepcopy
        #             # thread-safe
        #             # randIdx = rand.randint(0, args.num_clients)
        #             # randIdx = rand.randint(0, args.num_clients-1)
        #             # randIdx = client_id # fixed
        #             # if randIdx == client_id:
        #             if True:
        #                 item = itemData(my_input=copy.deepcopy(m_inputs), my_label=copy.deepcopy(m_labels))
        #                 if rand_counter <= client_data_len:
        #                     client_data.append(item)
        #                     rand_counter = rand_counter + 1
        #                 else:
        #                     is_need_random = False # break
        #                     break # end for

        #         # ----------------- random local dataset ------------------- #
        #         # stop randomizing the local dataset
        #         # double check....
        #         if len(client_data) >= client_data_len:
        #             is_need_random = False
        #             break # end while
        #     # ~general while

        #     data_end_time = time.time() # ending
        #     data_duration_time  = float(data_end_time - data_start_time) # in second
                
        #     logger.info("Client {} has client_data: len={}, duration={:.3f} seconds".format(client_id, 
        #                                                         len(client_data),
        #                                                         data_duration_time))

        # dataset_len = len(train_loader.dataset)
        # epoch_size = dataset_len//batch_size
        ######################################################################################################

        # set stepsize
        current_stepsize = list_stepsize_sequence[epoch]
        optimizer.set_stepsize(current_stepsize)

        current_sample_size = list_sample_sequence[epoch]

        logger.info("Client {} will run for {} iterations, epoch={}".format(client_id, current_sample_size, epoch))

        # training process
        # ref: https://stackoverflow.com/questions/825909/catch-only-some-runtime-errors-in-python
        # Sometimes, we got the CUDA out of memory running error.
        # And the training process is hang up or is being deadlock.
        start_epoch_time = time.time()
        try:
            train(args, model, device, None, 
                optimizer, epoch, 
                client_id, internal_queue_gradient,
                client_data)
        except RuntimeError as train_running_err:
            logger.error("Client {} encountered err={}".format(client_id, str(train_running_err)))
            os._exit(-1)
        #~training

        end_epoch_time = time.time()
        epoch_time_duration = float(end_epoch_time - start_epoch_time)


        # measure the duration...
        total_client_duration = total_client_duration + epoch_time_duration
        arr_duration[client_id] = total_client_duration

        logger.info("# Client {} has epoch_time_duration={:.3f} total_client_duration={:.3f}".format(client_id, epoch_time_duration,
                                                                                            total_client_duration))

        logger.info("Client {} starts the testing process # {}".format(client_id, epoch))
        test_accuracy, test_loss = test(args, 
                            model, 
                            device, 
                            test_loader, 
                            epoch, 
                            args.epochs, 
                            client_id)
        
        # add accuracy to client queue
        # for tracking later
        array_evalObj[client_id].get_test_accuracy().append(test_accuracy)

        # end_epoch_time = time.time()
        # epoch_time_duration = float(end_epoch_time - start_epoch_time)
        # logger.info("Client {} has test_accuracy={:.3f}, epoch={}, test_loss={:.3f}, epoch_duration={:.3f}".format(client_id, 
        #                                                                                                         test_accuracy*100.0, 
        #                                                                                                         epoch, test_loss,
        #                                                                                                         epoch_time_duration))

        logger.info("Client {} has test_accuracy={:.3f}, epoch={}, test_loss={:.3f}".format(client_id, 
                                                                                                                test_accuracy*100.0, 
                                                                                                                epoch, test_loss))

        # Add some asserts here...
        # https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values?rq=1
        if math.isnan(test_loss):
            logger.error("# Client {} has the test_loss equal to nan.".format(client_id))
            os._exit(-2) # stopping
        # ~asserts

        # testing process.
        # client does not need to run the testing process.
        # reduce time
        # test_accuracy = test(args, model, device, test_loader, epoch, args.epochs, client_id)
        # logger.info("Client {} has test_accuracy={:6f}, epoch={}".format(client_id, test_accuracy*100, epoch))


        ##### New configuration #####
        # Check for new global model for every iteration.
        # has_new_model = False
        # logger.info("Client {} wait for new global model, epoch={}".format(client_id, epoch))
        # while(has_new_model == False):
        #     has_new_model = current_global_model.has_new_global_model(block_index=client_id)
        #     if has_new_model == True:
        #         break
        #     else:
        #         time.sleep(0.01) # 10ms
        
        # ##### Have new model #####
        # if has_new_model == True:
        #     temp_w = current_global_model.get_global_model()

        #     addr_w = hex(id(temp_w))
        #     logger.debug("[client] addr_w={}, epoch={}".format(addr_w, epoch))

        #     # copy...
        #     serial.unravel_model_params(model, temp_w)

        #     my_model_id = current_global_model.get_uuid()
        #     # list_id.append(my_model_id)

        #     current_global_model.reset_value_at_block(block_index=client_id)
        #     logger.info("Client {} switches new global model at round={}".format(client_id, epoch))

        # ##### New configuration #####
        # logger.info("Client {} : epoch {} : [snapshot] EpochMap is {}".format(client_id, epoch, epochCounterMap))

        # is_waiting = True
        # asynchronous_round = 1 # fixed value
        # num_trial = 0
        # while (is_waiting == True):
        #     bool_is_synchronize = is_synchronize_needed(client_id=client_id,
        #                                                 d_asynchronous_round=asynchronous_round,
        #                                                 current_round=epoch)
        #     if bool_is_synchronize == True:
        #         # wait for a while
        #         logger.warn("Client {} is waiting for its neighbors: asynchronous_round={}".format(client_id, asynchronous_round))
        #         time.sleep(1.0) # 5ms
        #         logger.info("Client {} : [snapshot] EpochMap is {}".format(client_id, epochCounterMap))
        #         num_trial = num_trial + 1
        #         if num_trial > 20:
        #             logger.error("Client {} increase the epochCounterMap...".format(client_id))

        #             # increase
        #             # this constraint is not  very strict
        #             # sometimes, we can relax this constraint to make the program run smoothly.
        #             list_neighnors = neighborMap[client_id]
        #             for element_neighbor in list_neighnors:
        #                 epoch_mutex_queue.acquire()
        #                 # epochCounterMap[update_client_id] = epochCounterMap[update_client_id] + 1
        #                 try:
        #                     # print('Do some stuff')
        #                     epochCounterMap[element_neighbor] = epochCounterMap[element_neighbor] + 1
        #                 finally:
        #                     # mutex.release()
        #                     epoch_mutex_queue.release()

        #             # break
        #             is_waiting = False
        #             break
        #     else:
        #         is_waiting = False
        #         break

        ##### New synchronization method #####
        synchronized_memory[client_id] = epoch

        is_waiting = True
        # d_round = (3)
        d_round = (5)
        while (is_waiting == True and config.REQUIRE_SYNCHRONIZED_POINT2==misc.FLAG_ENABLE):
            bool_is_synchronize = is_synchronize_needed_method2(client_id=client_id,
                                                                current_epoch=epoch,
                                                                d_asynchronous_round=d_round)
            if bool_is_synchronize == True:
                # wait for a while
                # logger.warn("Client {} is waiting for its neighbors: asynchronous_round={}".format(client_id, asynchronous_round))
                time.sleep(0.1 * 5) # unit=100ms
                logger.warn("Client {}: current epoch {} : [snapshot] synchronized_memory is {}".format(client_id, 
                                                                                                        epoch, 
                                                                                                        synchronized_memory))
            else:
                is_waiting = False
                break


        # tricky
        # this is the most fucking code I've written...
        # sleep_base = 1.0 / 1000.0
        sleep_base = 1.0 / 5000.0
        lowest_round = get_round_method(client_id=client_id, current_epoch=epoch)
        fraction_dummy = 1.0
        try:
            fraction_dummy = float(epoch) / float(lowest_round)
        except Exception as expt_dummp:
            fraction_dummy = 1.151111111111 #### dummy value.

        # time.sleep(sleep_base * fraction_dummy)
        # ~tricky


    ##### training done #####
    # if args.save_model:
    #     # Pytorch model file.
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    # Set the magic epoch
    # mark that the client has finished its computation process
    # prevent error from synchronization
    logger.info("Client {} sets the finished_flag={}".format(client_id, misc.FLAG_FINISHED_EPOCH))
    synchronized_memory[client_id] = misc.FLAG_FINISHED_EPOCH
    # ~finished training

    logger.info("Client {} : Stop client with threadName={}".format(thread_name, client_id))


##### New configuration #####
# def sgd_worker(worker_name="worker_1", thread_id=0, worker_gradient=None, neightbors=None):
#     # Start 2 threads: sender + reciver
#     # worker_handler = []
#     # thread_receiver_name = "ThreadServ_" + str(thread_id)
#     # thread_server = threading.Thread(target=central_receiver, args=(thread_receiver_name, args,))
#     # worker_handler.append(thread_server)

#     thread_name = "ThreadWorker_" + str(thread_id)
#     logger.info("Create thread_name={}".format(thread_name))
#     # assume thread_name == client_id
#     internal_thread = threading.Thread(target=main_function, 
#             args=(worker_name, thread_id, args, thread_id, worker_gradient))

#     # Add thread into thread handler.
#     worker_handler.append(internal_thread)


#     # Start all threads
#     for i in range(args.num_clients):
#         worker_handler[i].start()


#     # Done
#     for i in range(args.num_clients):
#         worker_handler[i].join()

#     # thread_server.join()



##### Network topology #####
def periodic_generate_network_topology(thread_name="ThreadTopology", num_clients=5, args=None):
    global neighborMap
    global neighborEdgeMapWrite
    global neighborEdgeMapRead
    global stop_signal

    num_edge = args.num_edge
    logger.info("Thread {} will have num_edge={}".format(thread_name, num_edge))

    # Create list of queues, based on the number of edges
    network_counter = 0
    while (stop_signal == False):
        # 120 seconds
        time.sleep(5 * 60)

        # generate network
        for idx in range(num_clients):
            # random # of neighbors
            if num_edge == 0: # random edges
                num_neighbor = random.randint(num_clients//2+1, num_clients-1)
            else:
                # fixed num_edge
                num_neighbor = num_edge

            array_neighbors = []
            while (len(array_neighbors) < num_neighbor):
                neighbor = random.randint(0, num_clients-1)
                if neighbor != idx and (neighbor not in array_neighbors):
                    array_neighbors.append(neighbor)

            # Update neighborMap
            neighborMap[idx] = array_neighbors


            
        # neighborMap[0] = [1, 2, 3]
        # neighborMap[1] = [0, 2, 3]
        # neighborMap[2] = [0, 1, 4]
        # neighborMap[3] = [1, 4, 0]
        # neighborMap[4] = [2, 3]

        logger.info("# neighborMap is {}".format(neighborMap))

        list_edge = []
        for key in range(num_nodes):
            neighbors = neighborMap[key]
            temp_array = []
            for node in neighbors:
                queue_name = str(key) + "_" + str(node)
                # we only keep the edges which is unique
                # remove duplicate edge
                # safety
                if queue_name not in list_edge:
                    list_edge.append(queue_name)
                temp_array.append(queue_name)

            # Update queueEdgeName
            neighborEdgeMapWrite[key] = temp_array

        # Update queueEdgeName
        for key in range(num_nodes):
            neighbors = neighborMap[key]
            temp_array = []
            for node in neighbors:
                queue_name = str(node) + "_" + str(key)
                # list_edge.append(queue_name)
                temp_array.append(queue_name)

            neighborEdgeMapRead[key] = temp_array


        logger.info("# {} neighborMap is {}".format(network_counter, neighborMap))
        logger.info("# {} Queue name: len={}".format(network_counter, len(list_edge)))
        logger.info("# {} Queue name is {}".format(network_counter, list_edge))
        logger.info("# {} neighborEdgeMapWrite is {}".format(network_counter, neighborEdgeMapWrite))
        logger.info("# {} neighborEdgeMapRead is {}".format(network_counter, neighborEdgeMapRead))

        if len(queue_gradient) > 0:
            del queue_gradient[:]

        for idx in range(len(list_edge)):
            queue_name = list_edge[idx]
            internal_queue_gradient = memory_block.QueueGradient(queue_name=queue_name)
            queue_gradient.append(internal_queue_gradient)
    
        logger.info("# {} Created new internal gradient queues...".format(network_counter))

        # time.sleep(5 * 60) # 5 minutes
        network_counter = network_counter + 1

    ### Stopping
    logger.info("Thread: Network topology stopped.")


if __name__ == '__main__':
    ##### New configuration #####
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example --- LocalSGD Communication')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')

    # parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
    #                     help='input batch size for testing (default: 10000)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')

    # number of epochs.
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 15)')
    #~epoch.

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 1.0)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')


    # Add new parameters to run customized optimizer.
    parser.add_argument('--optim-method', type=str, default='SGD_fixround_sqrt_Decay',
                        choices=['Adam', 'SGD', 'SGD_Exp_Decay',
                                 'SGD_1t_Decay', 'SGD_1sqrt_Decay', 
                                 'SGD_fixround_t_Decay', 'SGD_fixround_sqrt_Decay', 'SGD_fixround_Exp_Decay',
                                 'SGD_None_Decay'], #### no decay, but need some customized actions
                        help='Which optimizer to use.')

    # initial stepsize
    # should not be large
    parser.add_argument('--eta0', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01).')
    # ~stepsize

    parser.add_argument('--decay_steps', type=int, default=1000,
                        help='Initial learning rate (default: 1000).')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Decay factor (default: 0.1).') 
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).') 
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum used in optimizer (default: 0.0).')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay used in optimizer (default: 0.0).')
    parser.add_argument('--milestones', nargs='*', default=[],
                        help='Used for SGD stagewise decay denoting when to decrease the step size, unit in iteration (default: []).')   
    parser.add_argument('--patience', type=int, default=10,
                        help='Used in ReduceLROnPlateau denoting number of epochs with no improvement after which learning rate will be reduced (default: 10).')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='Used in ReduceLROnPlateau for measuring the new optimum, to only focus on significant changes (default: 1e-4).')

    # dataset.
    # 0 --- mnist
    # 1 --- fashionmnist
    parser.add_argument('--dataset', type=int, default=misc.MNIST_DATASET,
                        help='Which dataset can be used for training?')


    parser.add_argument('--custom', type=int, default=0,
                        help='Customized or not?')

    parser.add_argument('--dynamic-batch', type=int, default=0, help='Dynamic batch?')
    #~customized optimizer.

    # number of clients
    parser.add_argument('--num-clients', type=int, default=2, metavar='N',
                        help='number of clients to train (default: 1)')

    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='privacy budget (default: 0.1).')

    parser.add_argument('--p-value', type=float, default=0.5,
                        help='exponential p (default: 0.5).')

    parser.add_argument('--num-edge', type=int, default=1, metavar='N',
                        help='number of edges that clients have (default: 2)')


    parser.add_argument('--need-constant-ss', type=int, default=0, metavar='N',
                        help='(default: 0)')

    parser.add_argument('--constant-ss', type=int, default=10, metavar='N',
                        help='(default: 10)')


    parser.add_argument('--total-iter', type=int, default=0, metavar='N',
                        help='(default: 0)')

    ##### New configuration ######
    # Add event name
    parser.add_argument('--event-name', type=str, 
                        default="localSGD", 
                        choices=['localSGD', 'EVENT'],
                        help='event name(default: 1)')

    args = parser.parse_args()
    # print("---------------------------------------")
    # print(args)
    logger.info("args={}".format(args))
    # print("---------------------------------------\n")

    # ref: https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    torch.manual_seed(args.seed)


    ##### New configuration #####
    for index in range(args.num_clients):
        evalObj = evalObject()
        array_evalObj.append(evalObj)

        # set the array of duration
        arr_duration.append(0.0)

    # Generate sample sequence
    config.DP_EXP_P = float(args.p_value) # p=?
    logger.info("p={}".format(config.DP_EXP_P))

    # load sigma from setting.json
    if config.EXPERIMENT_ID == "000000":
        config.DP_SIGMA = utils.get_sigma_by_p(p=config.DP_EXP_P) # p=?
        # config.DP_EPSILON = float(args.epsilon)
        total_iteration = utils.get_total_iteration()
        config.DP_EPSILON = float(args.epsilon) # eps=?
    elif config.EXPERIMENT_ID == "111111": # eps=?
        config.DP_EPSILON = float(args.epsilon)
        config.DP_SIGMA = utils.get_sigma_by_eps(eps=config.DP_EPSILON) # eps=?

    if args.total_iter == misc.FLAG_ZERO:
        total_iteration = utils.get_total_iteration()
        total_iteration = math.ceil( (total_iteration* 1.0) / (args.num_clients * 1.0))
        total_iteration = int(total_iteration)
        logger.info("# Split the total iterations into {} parts".format(args.num_clients))
    else:
        total_iteration = args.total_iter
    # Split the computation cost
    if total_iteration <= 0:
        logger.error("Total iteration is {}".format(total_iteration))
        os._exit(-1)

    logger.info("Total iteration (per client) is {}".format(total_iteration)) # K=?


    need_constant_ss = None
    if args.need_constant_ss == misc.FLAG_ENABLE:
        need_constant_ss = True
        logger.info("# Using constant local SGD")
        args.batch_size = (10)
    else:
        logger.info("# Using linearly increasing local SGD")
        # args.batch_size = (8)
        args.batch_size = (16)


    constant_ss = args.constant_ss

    logger.info("need_constant_ss {}, constant_ss={}".format(need_constant_ss, constant_ss)) # K=?

    # NEW:
    # if args.dataset == misc.CIFAR_10 and total_iteration < 100000: # 100K
    #     total_iteration = (100000)
    #     args.total_iteration = total_iteration # K=?
    #     logger.info("NEW: Total iteration is {}".format(total_iteration)) # K=?


    ##### New configuration #####
    # add the event name
    config.EXPERMENT_NAME = args.event_name # name=?
    config.EXPERMENT_NAME = config.EXPERMENT_NAME.upper()

    # double check
    if config.EXPERMENT_NAME not in config.EXPERMENT_NAME_ARR:
        logger.error("Setting error: wrong experiment name {}".format(config.EXPERMENT_NAME))
        os._exit(-1)
    else:
        logger.info("Setting: experiment_name is {}".format(config.EXPERMENT_NAME))
    # ~check

    # total_iteration = config.TOTAL_ITERATION # K=?
    if config.EXPERIMENT_ID == "000000":
        list_sample_sequence = utils.generate_sampling_linear_sampling_by_p(budget_iteration=total_iteration,
                                                                        need_eps=None,
                                                                        p_value=config.DP_EXP_P,
                                                                        need_constant_ss=need_constant_ss,
                                                                        constant_ss=constant_ss)
    elif config.EXPERIMENT_ID == "111111": # eps
        list_sample_sequence = utils.generate_sampling_linear_sampling_by_p(budget_iteration=total_iteration,
                                                                        need_eps=config.DP_EPSILON,
                                                                        p_value=config.DP_EXP_P,
                                                                        need_constant_ss=need_constant_ss,
                                                                        constant_ss=constant_ss)

    # generate stepsize
    list_stepsize_sequence = utils.get_list_stepsize(start_stepsize=args.eta0,
                                                    total_epoch=len(list_sample_sequence),
                                                    list_sampling=list_sample_sequence,
                                                    stepsize_method=3)

    temp_iter1 = []
    for index2 in range(len(list_sample_sequence)):
        temp_val = list_sample_sequence[0:index2+1]
        sum_so_far = sum(temp_val)
        temp_iter1.append(sum_so_far)

    list_increasing_iteration = copy.deepcopy(temp_iter1)


    # initializing N last elements
    # https://www.geeksforgeeks.org/python-get-last-n-elements-from-given-list/
    N_last_elelemt = 10
    # logger.info("Sample sequence: {}".format(list_sample_sequence))
    # logger.info("Sample stepsize: {}".format(list_stepsize_sequence))
    # logger.info("Sample iteration: {}".format(list_increasing_iteration))

    logger.info("Sample sequence: {}".format(list_sample_sequence[-N_last_elelemt:]))
    logger.info("Sample stepsize: {}".format(list_stepsize_sequence[-N_last_elelemt:]))
    logger.info("Sample iteration: {}".format(list_increasing_iteration[-N_last_elelemt:]))

    # important
    # update the number of rounds (or epochs)
    args.epochs = len(list_sample_sequence)
    logger.info("Number of epochs: {}".format(args.epochs))
    # os._exit(0)


    # # load sigma from setting.json
    # if config.EXPERIMENT_ID == "000000":
    #     config.DP_SIGMA = utils.get_sigma_by_p(p=config.DP_EXP_P) # p=?
    #     config.DP_EPSILON = float(args.epsilon)
    # else:
    #     config.DP_EPSILON = float(args.epsilon)
    #     config.DP_SIGMA = utils.get_sigma_by_eps(eps=config.DP_EPSILON) # eps=?
    if config.ENABLE_DP != misc.FLAG_DISABLE:
        logger.info("sigma={}".format(config.DP_SIGMA))
        logger.info("epsilon={}".format(config.DP_EPSILON))


    ##### New configuration #####
    # args.batch_size = int(args.batch_size * max(1, config.DP_EPSILON))
    # if args.batch_size > 256:
    #     args.batch_size = 256 # fixed max batch_size

    logger.info("batch_size={}".format(args.batch_size))


    # Init memory block
    # We will init global model once
    # After that, each worker just needs to work with its gradient_queue

    # device = torch.device("cpu")
    logger.info("Create global model, and share with other workers...")

    # https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    # Currently, we donot run the experiments with CIFAR-100
    assert(args.dataset >= misc.MNIST_DATASET)
    assert(args.dataset <= misc.CIFAR_10)
    # ~checking...

    if (args.dataset == misc.MNIST_DATASET) or (args.dataset == misc.FASHION_MNIST_DATASET) or (args.dataset == misc.KMNIST_DATASET):
        # temp_global_model = LeNet5().to(device) # letnet5
        temp_global_model = LeNet.myLeNet().to(device)
    elif args.dataset == misc.CIFAR_10:
        # AlexNet.AlexNet()
        temp_global_model = AlexNet.AlexNet().to(device)
    
    # show the model information: size, layers, input size, etc.
    # ref: https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
    if args.dataset != misc.CIFAR_10:
        summary(temp_global_model, (1,28,28))
    else:
        summary(temp_global_model, (3,32,32))


    # wait to see the model's information
    time.sleep(1) # 5 seconds

    # global_model = torch.rand(serial.ravel_model_params(temp_global_model).numel()).uniform_(0.001, 0.01)
    global_model = serial.ravel_model_params(temp_global_model, grads=False)
    # global_model = LeNet5()
    # summary(model, (1, 28, 28))
    # zero_gradients = torch.zeros(serial.ravel_model_params(global_model).size())
    current_global_model = memory_block.GlobalModel(num_block=args.num_clients, 
                                    model=global_model, init_val=None)
    # Update global memory block.
    current_global_model.update_model(new_global_model=global_model)
    #~global model

    ##### New configuration #####
    # for idx in range(args.num_clients):
    #     internal_queue_gradient = memory_block.QueueGradient()
    #     queue_gradient.append(internal_queue_gradient)
    
    # logger.info("Created new internal gradient queues...")

    ##### Set up neighnor map #####
    # static map
    # un-directed network totology

    # node_0 --- node_1 --- node_2
    #   |           |          |
    # ----------- node_3 --------
    #   |           |          |
    # ----------- node_4 --------


    # neighborMap[0] = [1, 3, 4]
    # neighborMap[1] = [0, 2, 3, 4]
    # neighborMap[2] = [1, 3, 4]
    # neighborMap[3] = [0, 1, 2]
    # neighborMap[4] = [0, 1, 2]

    # ring network topology
    neighborMap[0] = [1, 2]
    neighborMap[1] = [0, 3]
    neighborMap[2] = [0, 4]
    neighborMap[3] = [1, 4]
    neighborMap[4] = [2, 3]

    # mesh network topology
    # neighborMap[0] = [1, 2, 3, 4]
    # neighborMap[1] = [0, 2, 3, 4]
    # neighborMap[2] = [0, 1, 3, 4]
    # neighborMap[3] = [1, 2, 4, 0]
    # neighborMap[4] = [2, 3, 0, 1]

    # new network topology
    # neighborMap[0] = [1, 2, 3]
    # neighborMap[1] = [0, 2, 3]
    # neighborMap[2] = [0, 1, 4]
    # neighborMap[3] = [1, 4, 0]
    # neighborMap[4] = [2, 3]

    # generate network
    num_edge = args.num_edge
    num_clients = args.num_clients

    assert(args.num_edge >=0)
    assert(args.num_edge <= (args.num_clients-1))
    # assert(args.num_edge <= (args.num_clients))

    # ------------------- random network topology --------------------- #
    if config.TASK_ID == "7":
        logger.info("Start the process of random network topology")
        # reset the neighborMap
        # del neighborMap
        neighborMap = {}
        for key_element in range(num_clients):
            neighborMap[key_element] = []
        
        logger.info("# neighborMap is {}".format(neighborMap))
        transition_matrix = np.zeros(shape=(num_clients,num_clients))
        logger.info("# transition_matrix\n {}".format(transition_matrix))
        for idx in range(0,num_clients):
            # for ydx in range(idx+1, num_clients):
            ydx = idx + 1
            # if ydx < num_clients:
            #     # update the matrix
            #     print(idx, ydx)
            #     transition_matrix[idx][ydx] = 1.
            #     transition_matrix[ydx][idx] = 1.
            #     # break
            # elif ydx >= num_clients:
            if ydx >= num_clients:
                ydx=0

            # print(idx, ydx)
            transition_matrix[idx][ydx] = 1.
            transition_matrix[ydx][idx] = 1.

        logger.info("# transition_matrix2\n {}".format(transition_matrix))


        # os._exit(0)

        # list_key_element = []
        # for key_element in neighborMap:
        #     list_key_element.append(key_element)

        # list_key_element = []
        for idx in range(num_clients):
            # random # of neighbors
            # if num_edge == 0: # random edges
            #     num_neighbor = random.randint(num_clients//2+1, num_clients-1)
            # else:
            #     # fixed num_edge
            #     num_neighbor = num_edge
            num_neighbor = num_edge

            array_neighbors = []

            for m_idx in range(num_clients):
                if transition_matrix[idx][m_idx] == 1.0:
                    array_neighbors.append(m_idx)


            # check through the neighborMap
            # for key_element in neighborMap:
            #     copy_neighbor = neighborMap[key_element]
            #     if idx in copy_neighbor:
            #         array_neighbors.append(key_element)

            # while (len(array_neighbors) < num_neighbor):
            #     neighbor = random.randint(0, num_clients-1)
            #     if neighbor != idx and (neighbor not in array_neighbors):
            #         array_neighbors.append(neighbor)

            # print(array_neighbors)

            # Update neighborMap
            neighborMap[idx] = array_neighbors
            # list_key_element.append(idx)
        logger.info("End the process of random network topology")
    # ------------------- random network topology --------------------- #

    logger.info("neighborMap is {}".format(neighborMap))

    # interuption
    # os._exit(0)

    num_nodes = len(neighborMap.keys())

    if args.num_clients != num_nodes:
        logger.error("Need {} workers to run, but provide {}".format(num_nodes, args.num_clients))
        os._exit(-1)
    # ~static map

    # assert(args.num_edge >=0)
    # assert(args.num_edge <= (args.num_clients-1))

    # Create epochs map
    for idx in range(args.num_clients):
        epochCounterMap[idx] = 0
    
    logger.info("Created new epochCounterMap...")


    # Create list of queues, based on the number of edges
    list_edge = []
    for key in range(num_nodes):
        neighbors = neighborMap[key]
        temp_array = []
        for node in neighbors:
            queue_name = str(key) + "_" + str(node)
            # we only keep the edges which is unique
            # remove duplicate edge
            # safety
            if queue_name not in list_edge:
                list_edge.append(queue_name)
            temp_array.append(queue_name)

        # Update queueEdgeName
        neighborEdgeMapWrite[key] = temp_array

    # Update queueEdgeName
    for key in range(num_nodes):
        neighbors = neighborMap[key]
        temp_array = []
        for node in neighbors:
            queue_name = str(node) + "_" + str(key)
            # list_edge.append(queue_name)
            temp_array.append(queue_name)

        neighborEdgeMapRead[key] = temp_array


    logger.info("neighborMap is {}".format(neighborMap))
    logger.info("Queue name: len={}".format(len(list_edge)))
    logger.info("Queue name is {}".format(list_edge))
    logger.info("neighborEdgeMapWrite is {}".format(neighborEdgeMapWrite))
    logger.info("neighborEdgeMapRead is {}".format(neighborEdgeMapRead))

    for idx in range(len(list_edge)):
        queue_name = list_edge[idx]
        internal_queue_gradient = memory_block.QueueGradient(queue_name=queue_name)
        queue_gradient.append(internal_queue_gradient)
    
    logger.info("Created new internal gradient queues...")

    # Create synchronized shared memory
    # ref: https://www.geeksforgeeks.org/python-initialize-empty-array-of-given-length/
    synchronized_memory = [0] * args.num_clients


    ##### New configuration
    kwargs = {'num_workers': 0, 'pin_memory': False}
    # ~num_workers=0

    # load train dataset.
    # Dataloader...
    normalized_val = [0,0]
    if args.dataset == misc.MNIST_DATASET: # mnist
        dataset_loader = datasets.MNIST
        logger.info("[client] We are using dataset MNIST...")

        # hardcode
        normalized_val[0] = 0.1307
        normalized_val[1] = 0.3081

    elif args.dataset == misc.FASHION_MNIST_DATASET: # fashionMNIST
        dataset_loader = datasets.FashionMNIST
        logger.info("[client] We are using dataset FashionMNIST...")

        normalized_val[0] = 0.5
        normalized_val[1] = 0.5
        # normalized_val[0] = 0.1307
        # normalized_val[1] = 0.3081


    elif args.dataset == misc.KMNIST_DATASET: # EMNIST
        dataset_loader = datasets.KMNIST
        logger.info("[client] We are using dataset KMNIST...")

        normalized_val[0] = 0.5
        normalized_val[1] = 0.5
        # normalized_val[0] = 0.1307
        # normalized_val[1] = 0.3081

    elif args.dataset == misc.CIFAR_10:
        logger.info("[client] We are using dataset CIFAR10...")
        dataset_loader = datasets.CIFAR10

    # load dataset
    if args.dataset != misc.CIFAR_10:
            train_loader = torch.utils.data.DataLoader(
                dataset_loader('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((normalized_val[0],), (normalized_val[1],))
                    ])),
                batch_size=args.batch_size,
                # collate_fn=my_collate, # use custom collate function here
                shuffle=True, 
                **kwargs)
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = dataset_loader(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Create client_data
    client_data = []
    num_repeat=(2)
    logger.info("# Server: start to load dataset...")
    for repeat_idx in range(num_repeat):
        for batch_idx, data in enumerate(train_loader, 0):
            m_inputs, m_labels = data
            item = itemData(my_input=copy.deepcopy(m_inputs), my_label=copy.deepcopy(m_labels))
            client_data.append(item)

    logger.info("# Server: Load dataset with len={}".format(len(client_data)))
    ##### ~configuration

    # os._exit(0)

    start=datetime.now()
    start_process_time = time.time()
    # start the main process here....

    # Start server thread
    # thread_server = threading.Thread(target=central_server, args=("ThreadServ", args,))
    # thread_server.start()
    # time.sleep(3)       # waiting time
    
    # Start client thread
    worker_handler = []
    for client_id in range(args.num_clients):
        thread_name = "ThreadWorker_" + str(client_id)
        logger.info("Create thread_name={}".format(thread_name))
        # assume thread_name == client_id
        worker_gradient = queue_gradient[client_id]
        internal_thread = threading.Thread(target=main_function, 
                args=(thread_name, client_id, args, client_id, worker_gradient, client_data))

        # Add thread into thread handler.
        worker_handler.append(internal_thread)


    # Network topology
    if config.REQUIRE_NEW_NETWORK_TOPOLOGY == misc.FLAG_ENABLE:
        network_topology_thread = threading.Thread(target=periodic_generate_network_topology,
                args=("ThreadTopology", args.num_clients, args,))

    # Start all threads
    for i in range(args.num_clients):
        worker_handler[i].start()

    # Start network_topology_thread
    if config.REQUIRE_NEW_NETWORK_TOPOLOGY == misc.FLAG_ENABLE:
        network_topology_thread.start()
    #~network_topology_thread


    # Done
    for i in range(args.num_clients):
        worker_handler[i].join()

    stop_signal = True # stop

    # Network topology
    if config.REQUIRE_NEW_NETWORK_TOPOLOGY == misc.FLAG_ENABLE:
        network_topology_thread.join()
    # ~Network topology


    # thread_server.join()

    # close gradient queue
    # queue_gradient.close()
    # logger.info("Closed gradient queue...")

    # main_function(thread_name="thread_0", args=args, client_id=0)
    #~main process.

    total_time = datetime.now() - start
    end_process_time = time.time()
    process_time_duration = float(end_process_time - start_process_time)

    # logger.info("*** Total running time is {}".format(total_time))
    logger.info("*** Total running time is {:.3f} seconds -- {:.3f} minutes -- {:.3f} hours".format(process_time_duration, 
                                                                                                    process_time_duration/60.0,
                                                                                                    (process_time_duration/60.0) / 60.0)
                                                                                                )
    logger.info("*** Total number of gradient sending is {}".format(num_send_gradient))

    # New configuration
    logger.info("# Duration {}".format(arr_duration))
    min_duration = min(arr_duration)
    logger.info("# Duration_min {:.3f}".format(min_duration))
    avg_duration = sum(arr_duration) / len(arr_duration)
    logger.info("# Duration_avg {:.3f}".format(avg_duration))
    # ~configuration

    ##### New configuration #####
    arrayAccuracy = array_evalObj[0].get_test_accuracy()
    arr_sum_gradient = array_evalObj[0].get_gradient_norm()

    print(arrayAccuracy)            # accuracy
    # print(arr_sum_gradient)         # norm of gradients

    # dump result to logs file.
    # hashmap to dump data
    resultHashMap = {}
    # resultHashMap['optim_method'] = args.optim_method
    # resultHashMap['accuracy'] = arrayAccuracy
    # resultHashMap['total_epoch'] = str(args.epochs)
    # resultHashMap['eta0'] = str(args.eta0)
    # resultHashMap['alpha'] = str(args.alpha)
    # resultHashMap['momentum'] = str(args.momentum)
    # resultHashMap['weight_decay'] = str(args.weight_decay)
    # resultHashMap['nesterov'] = str(args.nesterov)
    # resultHashMap['milestones'] = str(args.milestones)
    # resultHashMap['batch_size'] = str(args.batch_size)
    # resultHashMap['gradient_norm'] = arr_sum_gradient

    ##### NEw configuration --- choose the best accuracy #####
    best_arrayAccuracy = None
    best_worker = 0
    best_accuracy = 0.0
    for idx in range(args.num_clients):
        worker_arrayAccuracy = array_evalObj[idx].get_test_accuracy() # accuracy=? worker=?
        len_acc = len(worker_arrayAccuracy)
        if best_accuracy < worker_arrayAccuracy[len_acc - 1]:
            best_accuracy = worker_arrayAccuracy[len_acc - 1]
            best_worker = idx
            best_arrayAccuracy = copy.deepcopy(worker_arrayAccuracy)

    logger.info("Client {} has the best accuracy {}".format(best_worker, best_arrayAccuracy))

    # get the last accuracy
    last_best_acc = best_arrayAccuracy[-1:]
    logger.info("# The (last) best accuracy {}".format(last_best_acc))
    logger.info("# Total iteration is {} \n".format(total_iteration))


    # log uuid
    log_uuid = my_random_string(15)

    ##### New configuration #####
    dataset_name = "mnist" # default
    if args.dataset == misc.MNIST_DATASET:
        dataset_name = "mnist"
    elif args.dataset == misc.FASHION_MNIST_DATASET: # fashionMNIST
        dataset_name = "fashionMNIST"
    elif args.dataset == misc.KMNIST_DATASET: # EMNIST
        dataset_name = "kmnist"
    elif args.dataset == misc.CIFAR_10:
        dataset_name = "cifar10"

    resultHashMap['exp_p'] = args.p_value # p=?
    resultHashMap['dataName'] = dataset_name
    resultHashMap['problem_type'] = 3 # non-convex
    resultHashMap['result'] = arrayAccuracy # test accuracy
    resultHashMap['list_sample_sequence'] = list_sample_sequence
    resultHashMap['eps_only'] = args.epsilon
    resultHashMap['exp_type_id'] = config.EXPERIMENT_ID
    resultHashMap['biased_ds'] = config.ENABLE_BIASED_DATASET # i.i.d
    resultHashMap['total_iteration'] = config.TOTAL_ITERATION # K=?

    resultHashMap['num_client'] = str(args.num_clients)
    resultHashMap['batch_size'] = str(args.batch_size)
    resultHashMap['total_epoch'] = str(args.epochs)
    resultHashMap['constant_ss'] = str(args.constant_ss)
    resultHashMap['need_constant_ss'] = str(args.need_constant_ss)
    

    resultHashMap['stepsize'] = "diminishing"
    resultHashMap['is_constant'] = 0 # eta_t
    resultHashMap['start_stepsize'] = str(args.eta0) # eta_0
    resultHashMap['has_dp_noise'] = str(config.HAS_DP_NOISE)
    resultHashMap['num_send_gradient'] = str(num_send_gradient)

    # local SGD
    resultHashMap['experiment_type'] = "localSGD"
    resultHashMap['best_acc'] = best_arrayAccuracy
    resultHashMap['best_worker'] = str(best_worker)

    ### New configuration
    # resultHashMap['array_evalObj'] = array_evalObj
    for idx in range(args.num_clients):
        worker_arrayAccuracy = array_evalObj[idx].get_test_accuracy() # accuracy=? worker=?
        node_name = 'node_' + str(idx)
        resultHashMap[node_name] = (copy.deepcopy(worker_arrayAccuracy))
    ### 
    resultHashMap['process_time'] = str(process_time_duration)

    net_string = "static"
    if config.REQUIRE_NEW_NETWORK_TOPOLOGY == misc.FLAG_ENABLE:
        net_string = "dynamic"

    # if args.num_edge == 0: random edge
    # if args.num_edge > 0: fixed edge
    resultHashMap['num_edge'] = str(args.num_edge)

    resultHashMap['net_topology'] = net_string

    # logging out
    type_dataset = "i.i.d"
    if config.ENABLE_BIASED_DATASET == misc.HETEROGENEOUS_DATASET:
        type_dataset = "heterogeneous"
    logger.info("Type of Dataset is iid? {}".format(type_dataset))

    # haha
    # check the directory status
    list_directory = ["event_logs", "event_eps", "event_pdf", "event_figure"]
    for directory in list_directory:
        directory_status = file_utils.check_directory_exist(directory=directory)
        if directory_status == False:
            logger.error("[Main] Directory {} does not exist...".format(directory))


    # (26+10)^{15} ~ 2.2107392e+23 is the total number of uuid.
    # net_string = "static"
    # if config.REQUIRE_NEW_NETWORK_TOPOLOGY == misc.FLAG_ENABLE:
    #     net_string = "dynamic"

    stringDataSet = config.TASK_ID + "_" + dataset_name + "_" + type_dataset + "_" + str(args.num_edge) + "_" + config.EXPERMENT_NAME + "_new_" + net_string
    if config.EXPERIMENT_ID == "000000": # p=?
        fileName = "logs/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".json"
    elif config.EXPERIMENT_ID == "111111": # eps=?
        fileName = "eps/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".json"

    # fileName = "logs/accuracy_" + dataset_name + ".json"
    logger.info("Export to file {}".format(fileName))
    with open(fileName, 'w+') as outfile:
        json.dump(resultHashMap, outfile)
    #~dumping.

    # Plot the chart of accuracy.
    # plot the line chart.
    ###################################### First figure
    fig, ax = plt.subplots()
    for idx in range(args.num_clients):
        worker_arrayAccuracy = array_evalObj[idx].get_test_accuracy() # accuracy=? worker=?
        geb_b30_x = np.arange(0, len(worker_arrayAccuracy), 1)
        # geb_b30_x = np.array(list_increasing_iteration)
        bar_val = pd.Series(data=worker_arrayAccuracy, index=geb_b30_x)
        ax.plot(geb_b30_x, bar_val, label="node "+ str(idx+1))

    plt.ylabel('Test accuracy')
    # plt.xlabel('# of iteration')
    plt.xlabel('# of round')
    # set title
    plt.title(dataset_name)
    plt.grid(True)
    plt.legend(loc=4, ncol=1)

    full_stringDataSet = config.TASK_ID + "_" + dataset_name + "_" + type_dataset + "_" + str(args.num_edge) + "_full_" + config.EXPERMENT_NAME + "_new_" + net_string + ".pdf"

    full_imgName = "pdf/accuracy_" + str(int(time.time())) + "_" + full_stringDataSet
    logger.info("[Final_0] Export full_chart to file_0 {}".format(full_imgName))
    try:
        fig.savefig(full_imgName)   # save the figure to file
    except Exception as expt_figure:
        logger.error("{}".format(str(expt_figure)))

    plt.close(fig)    # close the figure window



    ######################################## Second figure
    # for idx in range(args.num_clients):
    fig, ax = plt.subplots()
    worker_arrayAccuracy = array_evalObj[best_worker].get_test_accuracy() # accuracy=? worker=?
    geb_b30_x = np.arange(0, len(worker_arrayAccuracy), 1)
    # geb_b30_x = np.array(list_increasing_iteration)
    bar_val = pd.Series(data=worker_arrayAccuracy, index=geb_b30_x)
    ax.plot(geb_b30_x, bar_val, label="localSGD")

    # plt.xlabel('Iteration')
    plt.ylabel('Test accuracy')
    # plt.xlabel('# of iteration')
    plt.xlabel('# of round')
    # set title
    plt.title(dataset_name)
    plt.grid(True)
    plt.legend(loc=4, ncol=1)

    imgName = fileName.replace("json", "png") # json --> png
    imgName = imgName.replace("logs", "figure") # logs --> figure
    logger.info("[Final_1] Export chart to file_1 {}".format(imgName))

    # png
    try:
        fig.savefig(imgName)   # save the figure to file
    except Exception as expt_figure:
        logger.error("{}".format(str(expt_figure)))

    # pdf
    if config.EXPERIMENT_ID == "000000": # p=?
        fileName_pdf = "pdf/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".pdf"
    elif config.EXPERIMENT_ID == "111111": # eps=?
        fileName_pdf = "pdf/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".pdf"

    logger.info("[Final_2] Export chart to file_2 {}".format(fileName_pdf))
    try:
        fig.savefig(fileName_pdf)   # save the figure to pdf file
    except Exception as expt_figure:
        logger.error("{}".format(str(expt_figure)))
    
    plt.close(fig)    # close the figure window


    os._exit(0)
    ### Done...
