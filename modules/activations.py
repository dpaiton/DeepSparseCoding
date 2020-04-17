import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_picker(activation_function):
    if activation_function == "identity":
      return lambda x: x
    if activation_function == "relu":
      return F.relu
    if activation_function == "lrelu" or activation_function == "leaky_relu":
      return F.leaky_relu
    if activation_function == "lca_threshold":
      return lca_threshold
    assert False, ("Activation function " + activation_function + " is not supported!")

def lca_threshold(u_in, thresh_type, rectify, sparse_threshold, device="cpu"):
    u_zeros = torch.zeros_like(u_in, device=device)
    if thresh_type == "soft":
        if rectify:
            a_out = torch.where(torch.gt(u_in, sparse_threshold), u_in - sparse_threshold,
                u_zeros)
        else:
            a_out = torch.where(torch.ge(u_in, sparse_threshold), u_in - sparse_threshold,
                torch.where(torch.le(u_in, -sparse_threshold), u_in + sparse_threshold,
                u_zeros))
    elif thresh_type == "hard":
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
        assert False, ("Parameter thresh_type must be 'soft' or 'hard', not "+thresh_type)
    return a_out
