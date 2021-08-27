import torch
import os
import copy

# ref: https://github.com/ucla-labx/distbelief/blob/master/distbelief/utils/serialization.py

def ravel_model_params(model, grads=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    # global device
    # device = torch.device("cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda")

    m_parameter = torch.Tensor([0])
    m_parameter = m_parameter.to(device)

    for parameter in list(model.parameters()):
        if (grads == True) and (parameter.grad is not None):
            param_grad = parameter.grad.view(-1)
            # if param_grad is not None:
            m_parameter = torch.cat((m_parameter, param_grad))
        elif parameter.data is not None:
            param_data = parameter.data.view(-1)
            # if param_data is not None:
            m_parameter = torch.cat((m_parameter, param_data))

    return m_parameter[1:]


def unravel_model_params(model, parameter_update, grads=False):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() 
    and assigning the relevant params in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0 
    # keep track of where to read from parameter_update
    if grads == False:
        # data...
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
            current_index += numel
    else:
        # gradient...
        for parameter in model.parameters():
            numel = parameter.grad.numel()
            size = parameter.grad.size()
            parameter.grad.copy_(parameter_update[current_index:current_index+numel].view(size))
            current_index += numel


def compute_model_norm_different(m_local_model, m_current_model):
    # local model weights
    temp_local_model_weight = ravel_model_params(m_local_model, grads=False)

    # current model weights
    temp_current_model_weight = ravel_model_params(m_current_model, grads=False)

    # output
    # temp_local_output_weight = ravel_model_params(m_local_model, grads=False)

    # ref: https://pytorch.org/docs/stable/generated/torch.sub.html
    # temp_local_output_weight = (temp_local_model_weight - temp_current_model_weight)
    temp_local_output_weight = torch.sub(temp_local_model_weight, temp_current_model_weight)

    # ref: https://arxiv.org/pdf/1909.05020.pdf
    total_norm = temp_local_output_weight.norm(1)
    # total_norm = total_norm ** (1. / 2.0)

    return total_norm

