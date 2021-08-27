"""
Adapted from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
"""

import torch
from torch.optim import Optimizer
import math
from queue import Queue
import copy
import serialization as serial
import DP_config as config
import misc as misc


# my_queue = Queue(maxsize=10)

##################################################################################
# queue_max_size = (5)
# history_gradient_vec = Queue(maxsize=queue_max_size)

# # statistically information
# hashGradient = {}
# hashGradient['total'] = 0
# hashGradient['postive'] = 0
# hashGradient['negative'] = 0

# Ref: https://discuss.pytorch.org/t/runtimeerror-expected-object-of-backend-cpu-but-got-backend-cuda-for-argument-2-weight/38707
# If you are pushing tensors to the device, you have to reassign them.
# use_cuda = torch.cuda.is_available()
# torch.manual_seed(args.seed)
# device = torch.device("cpu")

# def string_format():
#     global hashGradient
#     return "total=" + str(hashGradient['total']) + " postive=" + str(hashGradient['postive']) + \
#         " negative="+ str(hashGradient['negative'])
##################################################################################


# 
def decayed_learning_rate(current_steps=0, decay_steps=1000, initial_learning_rate=0.1, alpha=0.1):
    step = min(current_steps, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha

    return initial_learning_rate * decayed


def linear_decayed_learning_rate(current_steps=0, decay_steps=1000, initial_learning_rate=0.1, alpha=0.1):
    beta=0.001

    step = min(current_steps, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.5 * step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta

    return initial_learning_rate * decayed

class SGDLRDecay(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    with several step size decay schemes (note that t starts from 1):
        1. 1/t decay: eta_t = eta_0 / (1 + alpha*t);
        2. 1/sqrt(t) decay: eta_t = eta_0 / (1 + alpha*sqrt(t));
        3. exponential decay: eta_t = eta_0 * (alpha**t);
        4. stagewise sgd: multiply eta_t by alpha at each milestone.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        scheme (str): the decay scheme, currently only supports {'exp', '1t',
            '1sqrt', 'stage'}.
        eta0 (float): initial learning rate.
        alpha (float): decay factor.
        milestones (list): a list denoting which time to decrease the stepsize.
        momentum (float, optional): momentum factor (default: 0).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        dampening (float, optional): dampening for momentum (default: 0).
        nesterov (bool, optional): enables Nesterov momentum (default: False).
    """

    def __init__(self, params, scheme, eta0, alpha, milestones=[],
                 momentum=0, dampening=0, weight_decay=0, nesterov=False, epoch_size=938, 
                 decay_steps=1000, model=None, is_customized=False):
        if eta0 < 0.0:
            raise ValueError("Invalid eta0 value: {}".format(eta0))
        if alpha < 0.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDLRDecay, self).__init__(params, defaults)

        self.eta0 = eta0
        self.alpha = alpha
        self.milestones = [int(x) for x in milestones]
        self.cur_round = 0
        self.cur_lr = eta0

        # update epoch size
        self.epoch_size = epoch_size
        self.decay_steps = decay_steps

        # mode
        self.model = model
        self.is_customized = is_customized

        # Define the function for computing the current step size for each decay.
        self.get_lr_func = None
        if scheme == 'exp':
            # self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: cur_lr * alpha
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 * math.pow(alpha, t)
        elif scheme == '1t':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 / (1.0 + alpha*t)
        elif scheme == '1sqrt':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 / (1.0 + alpha*(t**0.5))
        elif scheme == 'stage':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: cur_lr * alpha if t in milestones else cur_lr
        elif scheme == 'fixround_t':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 / (1.0 + alpha*(t//self.epoch_size)*self.epoch_size)
        elif scheme == 'fixround_sqrt':
            self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 / (1.0 + alpha*(((t//self.epoch_size)*self.epoch_size)**0.5))
        # elif scheme == 'fixround_exp':
        #     self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: eta0 * math.pow(alpha, (t//self.epoch_size)*self.epoch_size)
        # elif scheme == 'cosin':
        #     self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: decayed_learning_rate(current_steps=t, decay_steps=self.decay_steps, initial_learning_rate=eta0, alpha=alpha)
        # elif scheme == 'lincosin':
        #     self.get_lr_func = lambda cur_lr, t, eta0, alpha, milestones: linear_decayed_learning_rate(current_steps=t, decay_steps=self.decay_steps, initial_learning_rate=eta0, alpha=alpha)
        # elif scheme == 'fixed_eta':
        #     self.get_lr_func = None

        ##### New configuration #####
        temp_gradients = serial.ravel_model_params(self.model, grads=True)
        # temp_gradients = temp_gradients.to('cpu')
        self.accumulated_gradients = torch.zeros(temp_gradients.size())

        # convert tensor to cuda devices
        # self.accumulated_gradients = self.accumulated_gradients.to("cuda")

        # reset...
        # self.model.to("cuda")


    def __setstate__(self, state):
        super(SGDLRDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def reset_sum_gradient(self):
        self.accumulated_gradients.zero_()

    def get_sum_gradient(self):
        return self.accumulated_gradients
    
    def get_sum_gradient_norm(self):
        # for p in self.accumulated_gradients.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)

        total_norm = self.accumulated_gradients.norm(2)
        total_norm = total_norm ** (1. / 2.0)

        return total_norm

    ##### New configuration #####
    def set_stepsize(self, stepsize):
        self.cur_lr = stepsize

    def get_stepsize(self):
        return self.cur_lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.cur_round += 1
        # # checking...
        # if self.get_lr_func != None:
        #     self.cur_lr = self.get_lr_func(self.cur_lr, self.cur_round, self.eta0,
        #                                self.alpha, self.milestones)
        # else:
        #     # stupid, but meaningful...
        #     self.cur_lr = self.cur_lr #### fixed
        # ---------------------------------------------------------------------- #
        # limit the minimum stepsize
        # min = 1/10^7
        global_min_stepsize = float(1.0/(10000000))
        if self.cur_lr < global_min_stepsize:
            self.cur_lr = global_min_stepsize
        #~limit
        # ---------------------------------------------------------------------- #
        # if self.cur_round % 10 == 0:
        #     # print("[debug] iter={} --> lr={}".format(self.cur_round, self.cur_lr))
        #     gradient_norm = self.get_sum_gradient_norm()
        #     print("[debug] iter={} --> norm={}".format(self.cur_round, gradient_norm))

        
        #########################################################################
        # DP_CLIPPING_NORM = 0.1

        # checking the gradient
        DP_CLIPPING_NORM = config.DP_CLIPPING_NORM
        gradients = serial.ravel_model_params(self.model, grads=True)
        gradients = gradients.to('cpu')
        # gradients = torch.Tensor(gradients)
        # torch.nn.utils.clip_grad_norm_(gradients, DP_CLIPPING_NORM)

        # we only need to compute the gradient norms
        # when we apply the Gaussian noise
        # otherwise, ignore it

        # current_gradient_norm = gradients.norm(2)
        # current_gradient_norm = current_gradient_norm ** (1. / 2.0)
        if config.DP_EPSILON != config.MAGIC_EPSILON and config.ENABLE_DP == misc.FLAG_ENABLE:
            current_gradient_norm = gradients.norm(2)
            # current_gradient_norm = current_gradient_norm ** (1. / 2.0)
            gradients = gradients / max(1.0, current_gradient_norm/DP_CLIPPING_NORM)

        # sum of gradient
        # gradients = gradients.to('cpu')
        self.accumulated_gradients.add_(-self.cur_lr, gradients)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # gradient ratio
                # if gradient_radio != 0.0:
                #     d_p = p.grad.data * gradient_radio
                # else:

                # torch.nn.utils.clip_grad_norm_(p.grad, DP_CLIPPING_NORM)
                
                d_p = p.grad.data

                # Clip the gradient
                # we only need to compute the gradient norms
                # when we apply the Gaussian noise
                # otherwise, ignore it.

                # torch.nn.utils.clip_grad_norm_(d_p, DP_CLIPPING_NORM)
                # current_dp_norm = d_p.norm(2)
                # current_dp_norm = current_dp_norm ** (1. / 2.0)

                if config.DP_EPSILON != config.MAGIC_EPSILON and config.ENABLE_DP == misc.FLAG_ENABLE:
                    current_dp_norm = d_p.norm(2)
                    # current_dp_norm = current_dp_norm ** (1. / 2.0)
                    d_p = d_p / max(1.0, current_dp_norm/DP_CLIPPING_NORM)

                # if weight_decay != 0.0:
                #     d_p.add_(weight_decay, p.data)

                # if momentum != 0.0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                p.data.add_(-self.cur_lr, d_p)

        return loss