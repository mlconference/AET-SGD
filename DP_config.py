DP_SIGMA = (19.999999999999996)

DP_DELTA = (4.847394061450585e-08)

# DP_CLIPPING_NORM = 0.1 # C
DP_CLIPPING_NORM = 0.75 # C

DP_MOMENT_ORDER = 32

DP_EXP_P = (1.0) # p

DP_EPSILON = (0.05)

HAS_DP_NOISE = (0)

##### New configuration #####
ENABLE_DP = (0)


# iterations
# TOTAL_ITERATION = (63000)
TOTAL_ITERATION = (120000) # test


EXPERIMENT_ID = "000000" # p=?
# EXPERIMENT_ID = "111111" # eps=?
# EXPERIMENT_ID = "222222" # no-DP?

MAGIC_EPSILON = (9999)

##### New configuration ######
# 0 --- i.i.d
# 1 - heterogeneous
ENABLE_BIASED_DATASET = (0)


# p=?
CONFIG_JSON_FILE = "setting.json"

REQUIRE_SYNCHRONIZED_POINT = (0) # synchronization=?

REQUIRE_SYNCHRONIZED_POINT2 = (1) # synchronization=?


# dynamic network topology
REQUIRE_NEW_NETWORK_TOPOLOGY = (0)

REQUIRE_NOT_EQUAL_DATASET = (0)

##### New configuration --- event name #####
# EVENT
# LOCALSGD
EXPERMENT_NAME = "EVENT"

EXPERMENT_NAME_ARR = ["EVENT", "LOCALSGD"]



# split the dataset?
# total_point / num_client
REQUIRE_SPLIT_DATASET = (0) # no/yes


# Task name
# TASK_ID="0" # convergence
# TASK_ID="4" # vs. local SGD
# TASK_ID="6" # vs. number of iterations
TASK_ID = "7" # vs. number of workers
