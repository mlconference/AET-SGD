"""
Load the desired optimizer.
"""
import torch
import torch.optim as optim
from sgd_lr_decay import SGDLRDecay

###########################################################################
# history_gradient = None


############################################################################
# def ravel_model_params(model, grads=False):
#     """
#     Squash model parameters or gradients into a single tensor.
#     """
#     m_parameter = torch.Tensor([0])
#     for parameter in list(model.parameters()):
#         if (grads == True) and (parameter.grad is not None):
#             m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
#         elif parameter.data is not None:
#             m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
#     return m_parameter[1:]
##############################################################################


def load_optim(params, optim_method, eta0, alpha, milestones, nesterov,
               momentum, weight_decay, epoch_size, decay_steps, model=None,
               is_customized=False):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use, currently only supports {'Adam',
            'SGD', 'SGD_Exp_Decay', 'SGD_1t_Decay', 'SGD_1sqrt_Decay',
            'SGD_Stage_Decay', 'SGD_ReduceLROnPlateau'}.
        eta0: starting step size.
        alpha: decaying factor for various methods.
        milestones: used for SGD stage decay denoting when to decrease the
            step size, unit in iteration.
        nesterov: whether to use nesterov momentum (True) or not (False).
        momentum: momentum factor used in variants of SGD.
        weight_decay: weight decay factor.

    Outputs:
        an optimizer
    """

    # #########################################################################
    # # checking the gradient
    # # keep track of accumulated gradients so that we can send
    # global history_gradient

    # if (model is not None) and (model.parameters() is not None):
    #     gradients = ravel_model_params(model, grads=True)

    #     # update history gradient
    #     if history_gradient is not None:
    #         gradient_dot = torch.dot(history_gradient, gradients)
    #         gradient_norm = torch.norm(history_gradient)
    #         gradient_coherence = gradient_dot / gradient_norm
    #         print("{} gradient_coherence={}".format(gradient_coherence))
    #     else:
    #         print("history_gradient is None...")    
        
    #     history_gradient = gradients
    # else:
    #     print("Model is None...")
    # #########################################################################

    if optim_method == 'SGD' or optim_method == 'SGD_ReduceLROnPlateau':
        optimizer = optim.SGD(params=params, lr=eta0, momentum=momentum,
                              weight_decay=weight_decay, nesterov=nesterov)
    elif optim_method == 'Adam':
        optimizer = optim.Adam(params=params, lr=eta0,
                               weight_decay=weight_decay)
    elif optim_method.startswith('SGD') and optim_method.endswith('Decay'):
        if optim_method == 'SGD_Exp_Decay':
            scheme = 'exp'
        elif optim_method == 'SGD_1t_Decay':
            scheme = '1t'
        elif optim_method == 'SGD_1sqrt_Decay':
            scheme = '1sqrt'
        elif optim_method == 'SGD_Stage_Decay':
            scheme = 'stage'
        elif optim_method == 'SGD_fixround_t_Decay':
            scheme = 'fixround_t'
        elif optim_method == 'SGD_fixround_sqrt_Decay':
            scheme = 'fixround_sqrt'
        # elif optim_method == 'SGD_fixround_Exp_Decay':
        #     scheme = 'fixround_exp'
        # elif optim_method == 'SGD_cosin_Decay':
        #     scheme = 'cosin'
        # elif optim_method == 'SGD_lincosin_Decay':
        #     scheme = 'lincosin'
        # elif optim_method == 'SGD_None_Decay': # none_decay, need some customized actions
        #     scheme = 'fixed_eta'

        optimizer = SGDLRDecay( params=params, 
                                scheme=scheme, 
                                eta0=eta0,
                                alpha=alpha, 
                                milestones=milestones,
                                momentum=momentum, 
                                weight_decay=weight_decay,
                                nesterov=nesterov,
                                epoch_size=epoch_size,
                                decay_steps=decay_steps,
                                model=model,
                                is_customized=is_customized)
    else:
        raise ValueError("Invalid optimizer: {}".format(optim_method))

    return optimizer
