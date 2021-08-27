import pandas as pd
import numpy as np
import random
import pickle
import sys
import glob
import json
import os
import time


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.utils import column_or_1d
from sklearn.metrics import classification_report
from sklearn import preprocessing

# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# # matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import matplotlib.pyplot as plt
import seaborn as sns

# import config
# import utils

FIGURE_PATH = "figure/"
PDF_PATH = "pdf/"

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

# manually
# result_hashArray = {
#     '0.1': [],
#     '0.25': [],
#     '0.5': [],
#     '0.75': [],
#     '1.0': []
# }


# iteration_hashArray = {
#     '0.1': [],
#     '0.25': [],
#     '0.5': [],
#     '0.75': [],
#     '1.0': []
# }

# styledict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
# sns.set_style(style='whitegrid')

# plt.style.use('seaborn-paper')
# plt.grid(True)

list_key = []

expt_data_set = ""
problem_type = 0
problemType = ""

# list for all files
for fileName in glob.glob("eps/*.json", recursive=True):
    with open(fileName) as json_file:
        json_data = json.load(json_file)
        key_eps = str(json_data['eps_only'])
        list_key.append(key_eps)

        expt_data_set = json_data['dataName'] # data_name
        problem_type = int(json_data['problem_type'])

# convert list to set
unique_key = set(list_key)
unique_key = sorted(unique_key)
print(unique_key)

result_hashArray = {}
result_hashArray_Count = {}
iteration_hashArray = {}

for key_eps in unique_key:
    result_hashArray[key_eps] = []
    result_hashArray_Count[key_eps] = []
    iteration_hashArray[key_eps] = []

print(result_hashArray)
print(result_hashArray_Count)
print(iteration_hashArray)

global_min = 100000000
global_max = 0.0

# through all files
for fileName in glob.glob("eps/*.json", recursive=True):
    print(fileName)

    # load json data
    with open(fileName) as json_file:
        json_data = json.load(json_file)
        # print(json_data['dataName'])
        # print(json_data['result'])

        key_eps = str(json_data['eps_only'])
        result_arr = json_data['result']

        # check the accuracy or loss
        # if use_accuracy == False:
        for index in range(len(result_arr)):
            result_arr[index] = float(result_arr[index])
            # global_min = min(global_min, result_arr[index])

        # print(stepsize)
        # result_hashArray[stepsize].append(result_arr)
        final_accuracy = result_arr[len(result_arr) - 1] # final=?
        # final_accuracy = max(result_arr) # final=?

        # Get the result array.
        if len(result_hashArray[key_eps]) == 0:
            try:
                result_hashArray[key_eps].append(final_accuracy)
                result_hashArray_Count[key_eps].append(key_eps)

                # list_sample_sequence = json_data['list_sample_sequence']
                # iteration_hashArray[key_p].append(list_sample_sequence)

            except Exception as expt_bound:
                print("errr1")
                print(key_eps)
                print(expt_bound)
        else:
            try:
                result_hashArray[key_eps] = np.add(result_hashArray[key_eps], final_accuracy)
                result_hashArray_Count[key_eps].append(key_eps)
            except Exception as expt_bound2:
                print("errr2")
                print(key_eps)
                print(expt_bound2)

# do the averaging
key_item = result_hashArray.keys()
print(key_item)
for key in key_item:
    old_set = result_hashArray[key]
    temp_len = len(result_hashArray_Count[key])

    print("temp_len={}".format(temp_len))

    for index in range(len(old_set)):
        try:
            if temp_len > 1:
                result_hashArray[key][index] = result_hashArray[key][index]/float(temp_len)
        except Exception as epxt:
            print(epxt)
            print(key)
            print(result_hashArray[key][index])

# Show things
key_item = result_hashArray.keys()
for key_eps in key_item:
    print(str(key_eps) + ":" + str(result_hashArray[key_eps]))

# os._exit(0)


# prepare data
line = []
for key_eps in result_hashArray:
    try:
        line_0 = result_hashArray[key_eps][0]
        # line_0 = line_0[1: len(line_0)] # remove the first element
        # line_0.append(line_0[len(line_0) - 1])
    except Exception as expy_line0:
        print(expy_line0)
        line_0 = None

    # append to list
    line.append(line_0)

# find the global min
for each_line in line:
    # find lower bound
    local_min = (each_line) # float
    global_min = min(global_min, local_min)

    # find upper bound
    local_max = (each_line) # float
    global_max = max(global_max, local_max)


# plot the chart
label_arr = list(unique_key)

# ref: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/marker_reference.html
marker_list = ["2"]

fig, ax = plt.subplots()
for index in range(len(label_arr)):
    key_eps = label_arr[index] # key
    temp_line = line[index]
    array_temp_line = []
    array_temp_line.append(temp_line) # array

    copy_eps = []
    copy_eps.append(float(key_eps))
    # geb_b30_x = np.arange(0, len(array_temp_line))
    geb_b30_x = np.array(copy_eps)
    print(geb_b30_x)
    print(array_temp_line)

    bar_val = pd.Series(data=array_temp_line, index=geb_b30_x)

    # ax.plot(geb_b30_x, bar_val, label=("$\epsilon="+label_arr[index] + "$"), marker=".", markersize=20)
    ax.plot(geb_b30_x, bar_val, marker="*", markersize=12)

# show
# plt.xlabel('# of iteration')
plt.xlabel('privacy budget $\epsilon$')
plt.ylabel('Test accuracy')
# plt.legend(loc=4, ncol=1)

# set title
plt.title(expt_data_set)

plt.grid(True)
# plt.tight_layout()

print("global_min={}".format(global_min)) # global min
global_min = global_min - (0.05) * global_min
print("global_min2={}".format(global_min)) # global min2

print("global_max={}".format(global_max)) # global max
global_max = global_max + (0.027) * global_max
print("global_max2={}".format(global_max)) # global max2
if global_max >= 0.975:
    global_max = 1.0




plt.ylim(global_min, global_max) # gisette
# plt.ylim(0.6, 1.0) # others

# plt.show()

# print("Done...")
# os._exit(0)

if problem_type == 1: # strongly convex
    problemType = "strongly_convex"
elif problem_type == 2: # plain_convex
    problemType = "plain_convex"
elif problem_type == 3: # non-convex
    problemType = "non_convex"

timeStamp = str(int(time.time()))
file_name = FIGURE_PATH + "temp_eps_" + expt_data_set + "_" + problemType + ".png"
print("fileName={}".format(file_name))
fig.savefig(file_name)

# pdf
file_name_pdf = PDF_PATH + "temp_eps_" + expt_data_set + "_" + problemType + ".pdf"
print("fileName_pdf={}".format(file_name_pdf))
fig.savefig(file_name_pdf)

plt.close(fig)    # close the figure window

# Done
print("Done...")
os._exit(0)
