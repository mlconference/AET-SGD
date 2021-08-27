
# import warnings
import warnings
# filter warnings
# warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
import sys
import logging
import json
import DP_config as config

# --------------------------------- logging config -------------------------------- #
# create logger with 'framework'
logger = logging.getLogger('worker')
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s [%(threadName)-10.10s] [%(levelname)-8.8s]  %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)
# --------------------------------- logging config -------------------------------- #

# ---------------------------- worker configuration -------------------------- #
NUM_RUN_TIME = (1)
MAX_RUN_TIME = 5 # donot change
python_file = "main_SGD.py"
# ---------------------------- worker configuration -------------------------- #


################## Utils ######################
def get_list_eps(): # eps=?
    setting_json = "setting2.json"
    list_eps = [] # eps=?

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"] # p=?
            # json_eps = json_data["epsilon_value"] # eps=?
            # json_sigma = json_data["epsilon_value"] # sigma=?
            for item in json_p:
                search_key = float(item["epsilon_value"]) # eps=?
                list_eps.append(search_key)
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        # base_val = (-1)

    return list_eps


def get_new_list_eps(): # eps=?
    setting_json = "setting2.json"
    list_eps = [] # eps=?

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"] # p=?
            list_eps = json_data["eps"] # eps=?
            # json_sigma = json_data["epsilon_value"] # sigma=?
            # for item in json_p:
            #     search_key = float(item["epsilon_value"]) # eps=?
            #     list_eps.append(search_key)
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        # base_val = (-1)

    return list_eps


###############################################



def is_process_running():
    global python_file

    is_running = False
    command = " ps -ef | grep  -w " + python_file + "  | grep -v grep "
    logger.info(command)
    output = os.popen(command).read()
    if len(output) > 0:
        is_running = True
    else:
        is_running = False
    
    return is_running


def start_process(stepsize):
    command = "python " + python_file + "  " + str(stepsize)
    os.system(command)

# p=?
def start_process_2(p_fraction):
    # command = "python " + python_file + "  " + str(stepsize) + "  " + str(p_fraction)
    command = "CUDA_VISIBLE_DEVICES=2 python main_SGD.py --num-clients 5 --p-value {}".format(p_fraction)
    print(command)
    os.system(command)

# p=0.5, eps=?, K=120,000
def start_process_3(epsilon_value=1.0):
    # command = "python " + python_file + "  " + str(stepsize) + "  " + str(p_fraction)
    if epsilon_value != float(9999):
        command = "CUDA_VISIBLE_DEVICES=2 python main_SGD.py --num-clients 5 --epsilon {}".format(epsilon_value)
    else:
        command = "CUDA_VISIBLE_DEVICES=2 python main_SGD.py --num-clients 5 --epsilon {} --eta0 {}".format(epsilon_value, 0.01)

    print(command)
    os.system(command)

def start_process_222(num_edge=1):
    # command = "python " + python_file + "  " + str(stepsize) + "  " + str(p_fraction)
    command = "CUDA_VISIBLE_DEVICES=1,2 python main_SGD.py --num-clients 5 --dataset 0 --num-edge {}".format(num_edge)
    print(command)
    os.system(command)


# cifar10
def start_process_cifar10(num_edge=2):
    # command = "python " + python_file + "  " + str(stepsize) + "  " + str(p_fraction)
    command = "CUDA_VISIBLE_DEVICES=1,2 python3 main_SGD.py --num-clients 5 --dataset 3 --num-edge {}".format(num_edge)
    print(command)
    os.system(command)



# cifar10
# constant sample size
def start_process_cifar10_constant_ss(num_edge=2, const_ss=10):
    # command = "python " + python_file + "  " + str(stepsize) + "  " + str(p_fraction)
    command = "CUDA_VISIBLE_DEVICES=1,2 python3 main_SGD.py --num-clients 5 --dataset 3 --num-edge {} --need-constant-ss 1  --constant-ss {} ".format(num_edge, const_ss)
    print(command)
    os.system(command)

# epsilon=?
def start_process3(stepsize, epsilon_value):
    command = "python " + python_file + "  " + str(stepsize) + "  " + str(epsilon_value)
    os.system(command)


# def start_process3(stepsize, m_fraction, decaying_method):
#     command = "python " + python_file + "  " + str(stepsize) + "  " + str(m_fraction) + " " + str(decaying_method)
#     os.system(command)

def start_process4(stepsize, m_fraction, sampling_method):
    command = "python " + python_file + "  " + str(stepsize) + "  " + str(m_fraction) + " " + str(sampling_method)
    os.system(command)

def kill_all_process(process_name="python3"):
    command = " ps -ef | grep  -w " + python_file + "  | grep -v grep  | awk {'print $2'}"
    logger.info(command)
    output = os.popen(command).read()
    if len(output) > 0:
        # print(output)
        # is_running = True
        command2 = "kill -9 " + output
        os.system(command2)

if __name__== "__main__":
    # array_stepsize = ["0.1", "0.05", "0.01", "0.005", "0.001", "0.005", "0.0001"]

    # array_stepsize = ["0.01", "0.005", "0.001"] # fixed stepsize, used
    
    # array_stepsize = ["0.01"] # fixed

    # array_m_fraction = [0.25, 0.5, 0.75, 0.85, 1.0]
    # array_method = [0, 1, 2]

    # # ------------ fraction ------------ #
    array_stepsize = ["0.01"] # fixed (decaying) # libSVM
    # 0.001 (phishing)
    # array_p_fraction = [0.1, 0.25, 0.5, 0.75, 1.0]
    array_p_fraction = [0.01]
    # arr_sampling_method = [0, 1, 2, 3, 4]
    # arr_sampling_method = [1]

    array_eps_fraction = []
    # array_eps_fraction = get_new_list_eps()
    # array_eps_fraction = sorted(array_eps_fraction)

    # array_eps_fraction.append(9999)
    # array_eps_fraction.append(4)
    # array_eps_fraction.append(1)
    # print("eps={}".format(array_eps_fraction))

    if (len(sys.argv) == 2):
        NUM_RUN_TIME = int(sys.argv[1])

    logger.info("We will run file={} for {} time(s)".format(python_file, NUM_RUN_TIME))
    if NUM_RUN_TIME > MAX_RUN_TIME:
        # 
        logger.info("Exceed the maximum running times, max={}".format(MAX_RUN_TIME))
        os._exit(-1)
    
    # --------------- stepsize -------------- #
    # for stepsize in array_stepsize:
    #     for epoch in range(NUM_RUN_TIME):
    #         logger.info("Start process {}/{} with stepsize={}".format(epoch+1, NUM_RUN_TIME, stepsize))
    #         start_process(stepsize)
    # --------------- stepsize -------------- #
        

    # ------------ fraction ------------ #
    # array_p_fraction = sorted(array_p_fraction, reverse=True)
    # for fraction in array_p_fraction:
    #     for epoch in range(NUM_RUN_TIME):
    #         logger.info("Start process {}/{} with fraction={}".format(epoch+1, 
    #                                                                 NUM_RUN_TIME,
    #                                                                 fraction))
    #         # start_process(stepsize)
    #         start_process_2(fraction)
    # ------------ fraction ------------ #


    # ------------ epsilon? ------------ #
    # for epsilon in array_eps_fraction:
    #     for epoch in range(NUM_RUN_TIME):
    #         # if epsilon == 9999:
    #         logger.info("Start process {}/{} with stepsize={}, epsilon={}".format(epoch+1, 
    #                     NUM_RUN_TIME, array_stepsize[0], epsilon))
    #         # start_process(stepsize)
    #         start_process_3(epsilon)
   
    # ------------ fraction ------------ #


    # for sampling_method in arr_sampling_method:
    #     for epoch in range(NUM_RUN_TIME):
    #         logger.info("Start process {}/{} with stepsize={}, fraction={}, sampling_method={}".format(epoch+1, 
    #                         NUM_RUN_TIME, array_stepsize[0], 1.0, sampling_method))
    #         # start_process(stepsize)
    #         start_process4(array_stepsize[0], 1.0, sampling_method)
    
    # done code.

    ##### New configuration #####
    # num_edges = []
    # # # num_edges.append(0)
    # # for idx in range(0, 10):
    # #     num_edges.append(idx)
    
    # # num_edges = sorted(num_edges, reverse=True)
    # num_edges.append(2)

    # for num_edge in num_edges:
    #     for epoch in range(NUM_RUN_TIME):
    #         logger.info("Start process {}/{} with num_edge={}".format(epoch+1, NUM_RUN_TIME, num_edge))
    #         # start_process_222(num_edge=num_edge)
    #         start_process_cifar10(num_edge=num_edge)

    #         # waiting time
    #         time.sleep(3)


    # TASK 4
    constant_ss_arr = [10, 50, 100, 200, 500, 700, 1000]
    for constant_ss in constant_ss_arr:
        for epoch in range(NUM_RUN_TIME):
            logger.info("# STart process {}/{} with constant_ss={}".format(epoch+1, NUM_RUN_TIME,
                                                                            constant_ss))
            start_process_cifar10_constant_ss(num_edge=2, const_ss=constant_ss)
    # ~TASK 4

    os._exit(0)