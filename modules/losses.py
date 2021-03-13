import numpy as np
import torch

import DeepSparseCoding.utils.data_processing as dp


#def l2_flatness(z1, z2, z3, weight):
#    """
#    Minimized when a straight line can be drawn through [z1, z2, z3].
#    Extended from equations 8 and 12 in 
#    Chen, Paiton, Olshausen (2018) - The Sparse Manifold Transform
#    """
#    z_mat =  

def half_squared_l2(x1, x2):
    """
    Computes the standard reconstruction loss. It will average over batch dimensions.
    Keyword arguments:
        x1: Tensor with original input image
        x2: Tensor with reconstructed image for comparison
    Outputs:
        recon_loss: Tensor representing the squared l2 distance between the inputs, averaged over batch
    """
    dp.check_all_same_shape([x1, x2])
    reduc_dim = list(range(1, len(x1.shape))) # sum over reduc_dim, avg over batch
    squared_error = torch.pow(x1 - x2, 2.)
    recon_loss = torch.mean(0.5 * torch.sum(squared_error, dim=reduc_dim, keepdim=False))
    return recon_loss


def half_weight_norm_squared(weight_list):
    """
    Computes a loss that encourages each weight in the list of weights to have unit l2 norm.
    Keyword arguments:
        weight_list: List of torch variables
    Outputs:
        w_norm_loss: 0.5 * sum of (1 - l2_norm(weight))^2 for each weight in weight_list
    """
    w_norm_list = []
    for weight in weight_list:
        reduc_dim = list(range(1, len(weight.shape)))
        w_norm = torch.sum(torch.pow(1 - torch.sqrt(torch.sum(tf.pow(weight, 2.), axis=reduc_dim)), 2.))
        w_norm_list.append(w_norm)
    norm_loss = 0.5 * torch.sum(w_norm_list)
    return norm_loss


def weight_decay(weight_list):
    """
    Computes typical weight decay loss
    Keyword arguments:
        weight_list: List of torch variables
    Outputs:
        decay_loss: 0.5 * sum of weight^2 for each weight in weight_list
    """
    decay_loss = 0.5 * torch.sum([torch.sum(torch.pow(weight, 2.)) for weight in weight_list])
    return decay_loss


def l1_norm(latents):
    """
    Computes the L1 norm of for a batch of input vector
    This is the sparsity loss for a Laplacian prior
    Keyword arguments:
        latents: torch tensor of any shape, but where first index is always batch
    Outputs:
        sparse_loss: sum of abs of latents, averaged over the batch
    """
    reduc_dim = list(range(1, len(latents.shape)))
    sparse_loss = torch.mean(torch.sum(torch.abs(latents), dim=reduc_dim, keepdim=False))
    return sparse_loss


def trace_covariance(latents):
    """
    Returns loss that is the trace of the covariance matrix of the latents

    Keyword arguments:
        latents: torch tensor of shape [num_batch, num_latents] or [num_batch, num_channels, latents_h, latents_w]
    Outputs:
        loss
    """
    covariance = dp.covariance(latents) # [num_channels, num_channels]
    if latents.ndim == 4:
        num_batch, num_channels, latents_h, latents_w = latents.shape
        covariance = covariance / (latents_h * latents_w - 1.0)
    trace = torch.trace(covariance)
    target = torch.trace(torch.eye(covariance.size(0), device=trace.device)) # should = trace.size[0]
    return torch.norm(trace - target, p='fro')


def weight_orthogonality(weight, stride=1, padding=0):
    """
    Returns l2 loss that is minimized when the weight are orthogonal

    Keyword arguments:
        weight [torch tensor] layer weight, either fully connected or 2d convolutional
        stride [int] layer stride for convolutional layers
        padding [int] layer padding for convolutional layers

    Outputs:
        loss

    Note:
        Convolutional orthogonalization loss is based on
        Orthogonal Convolutional Neural Networks
        https://arxiv.org/abs/1911.12207
        https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks
    """
    w_shape = weight.shape
    if weight.ndim == 2: # fully-connected, [inputs, outputs]
        loss = torch.norm(torch.mm(weight.T, weight) - torch.eye(w_shape[1], device=weight.device))
    elif weight.ndim == 4: # convolutional, [output_channels, input_channels, height, width]
        out_channels, in_channels, in_height, in_width = w_shape
        output = torch.conv2d(weight, weight, stride=stride, padding=padding)
        out_height = output.shape[-2]
        out_width = output.shape[-1]
        target = torch.zeros((out_channels, out_channels, out_height, out_width),
            device=weight.device)
        center_h = int(np.floor(out_height / 2))
        center_w = int(np.floor(out_width / 2))
        target[:, :, center_h, center_w] = torch.eye(out_channels, device=weight.device)
        loss = torch.norm(output - target, p='fro')
    else:
        assert False, (f'weight ndim must be 2 or 4, not {weight.ndim}')
    return loss
