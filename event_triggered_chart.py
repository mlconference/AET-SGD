import pandas as pd
import numpy as np
import random
import pickle
import sys
import glob
import json
import os
import time

# --------------------------------- logging config -------------------------------- #
import logging

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

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDRegressor
# from sklearn.linear_model import SGDClassifier
# from sklearn.utils import column_or_1d
# from sklearn.metrics import classification_report
# from sklearn import preprocessing

# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# # matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

import matplotlib.pyplot as plt
# import seaborn as sns

fileName = "logs\\accuracy_1611806023_mnist_i.i.d_2_EVENT.json"
best_arrayAccuracy = []
expt_data_set = ""
num_edge = ""
EXPERMENT_NAME = ""
type_dataset = ""
EXPERIMENT_ID = ""

with open(fileName) as json_file:
    json_data = json.load(json_file)
    # key_eps = str(json_data['eps_only'])

    ##### New configuration #####
    # if float(key_eps) in plot_eps:
    #     list_key.append(key_eps)
    # else:
    #     continue

    expt_data_set = json_data['dataName'] # data_name
    problem_type = int(json_data['problem_type'])

    best_arrayAccuracy = json_data['best_acc']
    num_edge = json_data['num_edge']
    EXPERMENT_NAME = json_data['experiment_type'] # compared to local_SGD
    type_dataset = str(json_data['problem_type'])
    EXPERIMENT_ID = json_data["exp_type_id"]


# Plot the chart of accuracy.
# plot the line chart.
fig, ax = plt.subplots()
# # for idx in range(args.num_clients):
# #     worker_arrayAccuracy = array_evalObj[idx].get_test_accuracy() # accuracy=? worker=?
# #     # geb_b30_x = np.arange(0, len(worker_arrayAccuracy), 1)
# #     geb_b30_x = np.array(list_increasing_iteration)
# #     bar_val = pd.Series(data=worker_arrayAccuracy, index=geb_b30_x)
# #     ax.plot(geb_b30_x, bar_val, label="worker "+ str(idx+1))


# for idx in range(args.num_clients):
# worker_arrayAccuracy = array_evalObj[best_worker].get_test_accuracy() # accuracy=? worker=?
geb_b30_x = np.arange(1, len(best_arrayAccuracy)+1, 1) # round
# geb_b30_x = np.array(list_increasing_iteration)
bar_val = pd.Series(data=best_arrayAccuracy, index=geb_b30_x)
ax.plot(geb_b30_x, bar_val, label="event-triggered")

# # plt.xlabel('Iteration')
plt.ylabel('Test accuracy')
plt.xlabel('# of round')
# # set title
plt.title(expt_data_set)
plt.grid(True)
plt.legend(loc=4, ncol=1)

stringDataSet = expt_data_set + "_" + type_dataset + "_" + str(num_edge) + "_" + EXPERMENT_NAME

imgName = fileName.replace("json", "png") # json --> png
imgName = imgName.replace("event_logs", "event_figure") # logs --> figure
logger.info("[Final_1] Export chart to file_1 {}".format(imgName))

# # png
try:
    fig.savefig(imgName)   # save the figure to file
except Exception as expt_figure:
    logger.error("{}".format(str(expt_figure)))

# # pdf
if EXPERIMENT_ID == "000000": # p=?
    # fileName_pdf = "event_pdf/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".pdf"
    fileName_pdf = "pdf/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".pdf"
elif EXPERIMENT_ID == "111111": # eps=?
    fileName_pdf = "event_pdf/accuracy_" + str(int(time.time())) + "_" + stringDataSet + ".pdf"

logger.info("[Final_2] Export chart to file_2 {}".format(fileName_pdf))
try:
    fig.savefig(fileName_pdf)   # save the figure to pdf file
except Exception as expt_figure:
    logger.error("{}".format(str(expt_figure)))

plt.close(fig)    # close the figure window

os._exit(0)
### Done...
