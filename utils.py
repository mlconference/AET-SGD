import numpy as np
import time
import random
import string
import json
import math
from numpy import log as ln
import pandas as pd
import os
import DP_config as config

# ------------------------------ header files ------------------------- #

def randomString(stringLength=16):
    """ Generate a random string of letters and digits """
    
    letters = string.ascii_letters + string.digits
    
    return ''.join(random.choice(letters) for i in range(stringLength))


##### New configuration #####
def get_total_iteration():
    total_iteration = 0
    setting_json = config.CONFIG_JSON_FILE
    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"]
            total_iteration = int(json_data["K"])
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))

    return total_iteration


def get_sample_sequence_by_p(p=config.DP_EXP_P):
    key_p = float(p) # p=?
    setting_json = "setting.json"
    base_val = 1.0
    m = 1.0

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"]
            for item in json_p:
                search_key = float(item["p"]) # p=?
                if search_key == key_p:
                    base_val = float(item["base_val"])
                    m = float(item["m"])
                    # print(item)
                    break
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        base_val = (-1)

    return key_p, base_val, m

# get sigma value from p
def get_sigma_by_p(p=config.DP_EXP_P): # p=?
    key_p = float(p) # p=?
    setting_json = "setting.json"
    base_val = 1.0
    m = 1.0
    sigma = 1.0

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"] # p=?
            for item in json_p:
                search_key = float(item["p"]) # p=?
                if search_key == key_p:
                    # base_val = float(item["base_val"])
                    # m = float(item["m"])
                    sigma = float(item["sigma_value"]) # sigma=?
                    # print(item)
                    break
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        base_val = (-1)

    return sigma


# get sigma value from eps
def get_sigma_by_eps(eps=config.DP_EPSILON): # eps=?
    key_eps = float(eps) # eps=?
    setting_json = "setting2.json"
    base_val = 1.0
    m = 1.0
    sigma = 1.0

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"] # p=?
            for item in json_p:
                search_key = float(item["epsilon_value"]) # eps=?
                if search_key == key_eps:
                    # base_val = float(item["base_val"])
                    # m = float(item["m"])
                    sigma = float(item["sigma_value"]) # sigma=?
                    # print(item)
                    break
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        base_val = (-1)

    return sigma


def get_sample_sequence_by_eps(eps=config.DP_EPSILON): # eps=?
    key_eps = float(eps) # eps=?
    setting_json = "setting2.json"
    # CONFIG_JSON_FILE
    # setting_json = myconfig.CONFIG_JSON_FILE
    base_val = 1.0
    m = 1.0
    key_p = 1.0

    try:
        with open(setting_json) as json_file:
            json_data = json.load(json_file)
            json_p = json_data["p_value"] # p=?
            # json_eps = json_data["epsilon_value"] # eps=?
            # json_sigma = json_data["epsilon_value"] # sigma=?
            for item in json_p:
                search_key = float(item["epsilon_value"]) # eps=?
                if search_key == key_eps:
                    base_val = float(item["base_val"])
                    m = float(item["m"])
                    key_p = float(item["p"])
                    # print(item)
                    break
    except Exception as expt_json_fail:
        print("err: "+ str(expt_json_fail))
        base_val = (-1)

    return key_p, base_val, m


def generate_sampling_linear_sampling_by_p(budget_iteration=10000, need_eps=None, p_value=0.1,
                                            need_constant_ss=None, constant_ss=10):
    #-----------------------------------------------------#                                
    # function f(iter_round) = N * q * (i+m)^p.
    #-----------------------------------------------------#
    list_sampling = []

    iter_round = 1
    
    # load information from json file...
    if need_constant_ss == None:
        if need_eps == None:
            key_p, base_val, m = get_sample_sequence_by_p(p=p_value) #  p=?
            print("# key_p={}".format(key_p))
            # print(base_val)
            print("# base_val={}".format(base_val))
            # print(m)
            print("# m={}".format(m))
        else:
            my_eps = float(need_eps) # eps=?
            key_p, base_val, m = get_sample_sequence_by_eps(eps=my_eps) # eps=?
            # print(key_p)
            # print(base_val)
            # print(m)

            print("key_p2={}".format(key_p))
            print("base_val2={}".format(base_val))
            print("m2={}".format(m))
    else:
        base_val = constant_ss
        m = 0
        key_p = 1

        print("base_val={}".format(base_val))

    # validate
    if base_val < 0:
        print("err: cannot parse setting.json")
        os._exit(-1)

    if need_constant_ss == None:
        base_val = (10)
        # base_val = (10//2) # for higher accuracy
        m = (0)
        key_p = (1)

        # logging out
        print("Reconfig the parameters:")
        print("# key_p={}".format(key_p))
        # print(base_val)
        print("# base_val={}".format(base_val))
        # print(m)
        print("# m={}".format(m))
        print("# event={}".format(config.EXPERMENT_NAME))

    while(True):
        if need_constant_ss == None:
            # curent_sampling = math.ceil(base_val*(((iter_round) + m)**key_p))
            # curent_sampling = math.ceil(base_val*(((iter_round) + m)**key_p))
            # base_val = (10)
            # base_val = (10//2) # for higher accuracy
            # m = (0)
            # key_p = (1)

            curent_sampling = math.ceil(base_val*(((iter_round))**key_p))
        else:
            curent_sampling = math.ceil(base_val*(((iter_round) + m)**key_p))



        assert(curent_sampling >= 1)

        num_client = (1)
        curent_sampling_by_client = curent_sampling * num_client
        if curent_sampling_by_client < budget_iteration:
            budget_iteration = budget_iteration - curent_sampling_by_client
            list_sampling.append(curent_sampling)
        else:
            if budget_iteration % num_client == 0:
                list_sampling.append(budget_iteration//num_client)
            else:
                list_sampling.append(budget_iteration//num_client + 1)

            # terminate    
            break

        # increase iter_round
        if need_constant_ss == None:
            iter_round = iter_round + 1


    return list_sampling


def get_list_stepsize(start_stepsize=0.1,
                    total_epoch=10, 
                    list_sampling=None, 
                    stepsize_method=1):
    list_stepsize = []
    current_stepsize = start_stepsize

    if config.EXPERMENT_NAME != "EVENT":
        b = float(0.01) # non-convex
        # 10^{-3}
    else:
        b = float(0.00001) # event-triggered (e = 10^-5)
        # 10^{-5}
    #~new config.
    print("Setting: b={}".format(b))

    my_num_client = (1) # fixed value
    for index in range(total_epoch):
        temp_arr = list_sampling[0:index]
        num_sampling = sum(temp_arr)
        current_t = num_sampling * my_num_client

        if config.EXPERMENT_NAME != "EVENT":
            current_stepsize = start_stepsize * 1.0 / float(1.0 + b * math.sqrt(current_t))
        else:
            current_stepsize = start_stepsize * 1.0 / float(1.0 + b * (current_t))

        # log out
        # print("index={}, sr = {}, sampling={}, current_t={}, lr={}".format(index, 
        #                                                                 list_sampling[index],
        #                                                                 num_sampling, 
        #                                                                 current_t, 
        #                                                                 current_stepsize))

        list_stepsize.append(current_stepsize)
    
    return list_stepsize


if __name__ == '__main__':
    ##### Test epsilon #####
    eps_val = 9999
    sigma = get_sigma_by_eps(eps=eps_val)
    print("sigma={}".format(sigma))

    key_p, base_val, m = get_sample_sequence_by_eps(eps=eps_val)
    print("key_p2={}".format(key_p))
    print("base_val2={}".format(base_val))
    print("m2={}".format(m))

    list_sampling = generate_sampling_linear_sampling_by_p(budget_iteration=120000, need_eps=eps_val, p_value=0.5)
    print(list_sampling)
    print("len2 = {}".format(len(list_sampling)))

    ##### Test p #####
    # p_val = 0.01
    # sigma = get_sigma_by_p(p=p_val)
    # print("sigma={}".format(sigma))

    # key_p, base_val, m = get_sample_sequence_by_p(p=p_val)
    # print("key_p={}".format(key_p))
    # print("base_val={}".format(base_val))
    # print("m={}".format(m))

    # list_sampling = generate_sampling_linear_sampling_by_p(budget_iteration=120000, need_eps=None, p_value=p_val)
    # print(list_sampling)
    # print("len2 = {}".format(len(list_sampling)))


    ### Done
    os._exit(0)
