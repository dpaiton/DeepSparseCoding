import torch
import torch.nn as nn

def lca_threshold(u_in, thresh_type, rectify, sparse_threshold):
    u_zeros = torch.zeros_like(u_in)
    if thresh_type == 'soft':
        if rectify:
            a_out = torch.where(torch.gt(u_in, sparse_threshold), u_in - sparse_threshold,
                u_zeros)
        else:
            a_out = torch.where(torch.ge(u_in, sparse_threshold), u_in - sparse_threshold,
                torch.where(torch.le(u_in, -sparse_threshold), u_in + sparse_threshold,
                u_zeros))
    elif thresh_type == 'hard':
        if rectify:
            a_out = torch.where(
                torch.gt(u_in, sparse_threshold),
                u_in,
                u_zeros)
        else:
            a_out = torch.where(
                torch.ge(u_in, sparse_threshold),
                u_in,
                torch.where(
                    torch.le(u_in, -sparse_threshold),
                    u_in,
                    u_zeros))
    else:
        assert False, (f'Parameter thresh_type must be "soft" or "hard", not {thresh_type}')
    return a_out

def activation_picker(activation_function):
    if activation_function == 'identity':
      return nn.Identity()
    if activation_function == 'relu':
      return nn.ReLU()
    if activation_function == 'lrelu' or activation_function == 'leaky_relu':
      return nn.LeakyReLU()
    assert False, (f'Activation function {activation_function} is not supported.')
