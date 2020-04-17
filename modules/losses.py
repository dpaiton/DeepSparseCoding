import torch

import PyTorchDisentanglement.utils.data_processing as dp


def half_squared_l2(x1, x2):
    """
    Computes the standard reconstruction loss. It will average over batch dimensions.
    Args:
        x1: Tensor with original input image
        x2: Tensor with reconstructed image for comparison
    Returns:
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
    Args:
        weight_list: List of torch variables
    Returns:
        w_norm_loss: 0.5 * sum of (1 - l2_norm(w))^2 for each w in weight_list
    """
    w_norm_list = []
    for w in weight_list:
        reduc_dim = list(range(1, len(w.shape)))
        w_norm = torch.sum(torch.pow(1 - torch.sqrt(torch.sum(tf.pow(w, 2.), axis=reduc_dim)), 2.))
        w_norm_list.append(w_norm)
    norm_loss = 0.5 * torch.sum(w_norm_list)
    return norm_loss


def weight_decay(weight_list):
    """
    Computes typical weight decay loss
    Args:
        weight_list: List of torch variables
    Returns:
        decay_loss: 0.5 * sum of w^2 for each w in weight_list
    """
    decay_loss = 0.5 * torch.sum([torch.sum(torch.pow(w, 2.)) for w in weight_list])
    return decay_loss


def l1_norm(latents):
    """
    Computes the L1 norm of for a batch of input vector
    This is the sparsity loss for a Laplacian prior
    Args:
        latents: torch tensor of any shape, but where first index is always batch
    Returns:
        sparse_loss: sum of abs of latents, averaged over the batch
    """
    reduc_dim = list(range(1, len(latents.shape)))
    sparse_loss = torch.mean(torch.sum(torch.abs(latents), dim=reduc_dim, keepdim=False))
    return sparse_loss
