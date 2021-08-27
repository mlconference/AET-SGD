import torch
import numpy as np
import os


# ref: https://arxiv.org/pdf/1909.05020.pdf

STEPSIZE_EPSILON = float(10) ** (-5)

# beta_t
def get_beta_t_stepsize_by_iteration(current_iteration=1, initial_stepsize=0.1):
    fraction_1 = (STEPSIZE_EPSILON * current_iteration + 1)
    fraction_1 = fraction_1 ** (0.1)

    beta_t = initial_stepsize / fraction_1

    return beta_t

# alpha_t
def get_alpha_t_stepsize_by_iteration(current_iteration=1, initial_stepsize=0.1):
    fraction_1 = (STEPSIZE_EPSILON * current_iteration) + 1
    alpha_t = initial_stepsize / fraction_1

    return alpha_t
