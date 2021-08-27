import numpy as np
import random
from threading import Thread, Lock
import threading
import logging
# from multiprocessing import Queue
import multiprocessing

# ref: https://pymotw.com/3/queue/
import queue
# # Purpose:	Provides a thread-safe FIFO implementation
# The queue module provides a first-in, first-out (FIFO) data 
# structure suitable for multi-threaded programming. 
# It can be used to pass messages or other data between 
# producer and consumer threads safely. Locking is handled 
# for the caller, so many threads can work with 
# the same Queue instance safely and easily. 
# The size of a Queue (the number of elements it contains) 
# may be restricted to throttle memory usage or processing.

# q = queue.Queue()

import copy
import time

import utils

# ------------------------------- header files ------------------------- #

FLAG_WRITTEN_UPDATED = 1
FLAG_WRITTEN_NOT_UPDATED = 0
FLAG_BLOCK_NO_VALUE = None

# Model flag.
FLAG_HAS_NEW_MODEL = 1
FLAG_HAS_OLD_MODEL = 0

# create logger
logger = logging.getLogger('framework')

class MemoryBlock:
    def __init__(self, num_block=2, mermory_type=None):
        # Init shared memory.
        self.num_block = num_block
        self.shared_mem_weight = np.empty(num_block, dtype=mermory_type)
        self.shared_mem_b = np.empty(num_block, dtype=float)
        self.write_flag = np.empty(num_block)

        # create mutex
        self.mutex_mem = Lock()

        # At first, all blocks have been not update.
        for index in range(self.num_block):
            self.write_flag[index] = FLAG_WRITTEN_NOT_UPDATED

        for index in range(self.num_block):
            self.shared_mem_weight[index] = FLAG_BLOCK_NO_VALUE
            self.shared_mem_b[index] = 0



    def update_gradient_at_block(self, gradient_udate, b_update, block_index=0):
        self.mutex_mem.acquire()

        try:
            self.shared_mem_weight[block_index] = gradient_udate
            self.shared_mem_b[block_index] = b_update
            self.write_flag[block_index] = FLAG_WRITTEN_UPDATED

            logger.info("Update gradient at block_index={}".format(block_index))
        finally:
            self.mutex_mem.release()

    
    def has_all_new_update(self):
        is_all_new_update = True
        for index in range(self.num_block):
            if self.write_flag[index] == FLAG_WRITTEN_NOT_UPDATED:
                is_all_new_update = False
                break

        return is_all_new_update


    def get_gradient_at_block(self, block_index=0):
        logger.info("Get gradient at block_index={}".format(block_index))

        if block_index < self.num_block:
            return self.shared_mem_weight[block_index], self.shared_mem_b[block_index]
        return None, None


    def reset_gradient_at_block(self, block_index=0):
        self.mutex_mem.acquire()

        try:
            self.shared_mem_weight[block_index] = FLAG_BLOCK_NO_VALUE
            self.shared_mem_b[block_index] = FLAG_BLOCK_NO_VALUE
            self.write_flag[block_index] = FLAG_WRITTEN_NOT_UPDATED

            logger.info("Reset gradient at block_index={}".format(block_index))
        finally:
            self.mutex_mem.release()



class GlobalModel:
    def __init__(self, num_block=2, model=None, init_val=None, rand_val=False):
        self.num_block = num_block
        self.global_weight =  copy.deepcopy(model)
        # if init_val is not None:
        #     self.global_weight = np.full((model_dimension,1), init_val)
        #     self.global_b = init_val
        # else:
        #     # default values.

        #     min_val = 0.75
        #     max_val = 0.75
            
        #     # randomize values.
        #     self.global_weight = np.full((model_dimension,1), min_val)
        #     self.global_b = min_val

        #     if rand_val == True:
        #         for randIndex in range(model_dimension):
        #             self.global_weight[randIndex,:] = np.random.uniform(min_val, max_val)
                
        #         # random for b.
        #         self.global_b = np.random.uniform(min_val, max_val)

        # create num_block flags.
        self.read_flag = np.empty(self.num_block)

        # At first, all blocks have been not update.
        for index in range(self.num_block):
            self.read_flag[index] = FLAG_HAS_OLD_MODEL


        # genrate an uuid for global model
        self.uuid = utils.randomString()


    def get_global_model(self):
        return self.global_weight


    def get_uuid(self):
        return self.uuid


    # for tracking
    def get_global_model_id(self):
        return hash(self.global_weight)

    def get_model_info_at_block(self, block_index=0):
        return self.read_flag[block_index]

    
    def has_new_global_model(self, block_index=0):
        if self.read_flag[block_index] == FLAG_HAS_NEW_MODEL:
            return True

        return False

    
    def reset_value_at_block(self, block_index=0):
        self.read_flag[block_index] = FLAG_HAS_OLD_MODEL


    def notify_new_model(self):
        for index in range(self.num_block):
            self.read_flag[index] = FLAG_HAS_NEW_MODEL


    def update_model(self, new_global_model):
        self.global_weight = copy.deepcopy(new_global_model)
        # self.global_b = copy.deepcopy(new_global_b)

        # genrate an uuid for global model
        self.uuid = utils.randomString()


# ------------------------------------ adapt to queue ------------------------------- #
class GradientInformation:
    def __init__(self, gradient_sum, local_round, client_id):
        self.gradient_sum = copy.deepcopy(gradient_sum)
        self.local_round = copy.deepcopy(local_round)
        self.client_id = copy.deepcopy(client_id)


    def get_gradient_sum(self):
        return self.gradient_sum

    
    # def get_b_sum(self):
    #     return self.b_sum

    def get_local_round(self):
        return self.local_round

    def get_client_id(self):
        return self.client_id


class QueueGradient:
    def __init__(self, queue_name):
        self.q_gradient = queue.Queue()
        self.mutex_queue = Lock()
        self.queue_name = queue_name


    def enqueue(self, gradInfo):
        self.mutex_queue.acquire()
        try:
            # self.q_gradient.put(copy.deepcopy(gradInfo)) # new memory block...
            self.q_gradient.put(gradInfo) # new memory block...
            # time.sleep(0.01)
        finally:
            self.mutex_queue.release()


    def is_empty(self):
        is_empty = False
        self.mutex_queue.acquire()
        try:
            is_empty = self.q_gradient.qsize() == 0
            # time.sleep(0.01)
        finally:
            self.mutex_queue.release()
        
        # default is False
        return is_empty

    # queue size
    def get_internal_queue_size(self):
        return self.q_gradient.qsize()
        

    def dequeue(self):
        gradInfo = None
        self.mutex_queue.acquire()
        try:
            gradInfo = self.q_gradient.get()
            # time.sleep(0.01)
        finally:
            self.mutex_queue.release()

        # return
        return gradInfo

    def get_queue_name(self):
        # 0_1: client_0 sends the sum of gradients to client_1
        return self.queue_name

################### end of file ###################