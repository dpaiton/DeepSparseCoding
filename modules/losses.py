import tensorflow as tf

import DeepSparseCoding.utils.tensor_utils


def half_squared_l2(original, reconstruction):
    """
    Computes the standard reconstruction loss. It will average over batch dimensions.
    Args:
        original: Tensor with original input image
        reconstruction: Tensor with reconstructed image for comparison
    Returns:
        recon_loss: Tensor representing the squared l2 distance between the inputs, averaged over batch
    """
    tensor_utils.check_all_same_shape([original, reconstruction])
    reduc_dim = list(range(1, original.shape)) # sum over reduc_dim, avg over batch
    recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
        tf.square(original - reconstruction),
        axis=reduc_dim))
    return recon_loss


def half_weight_norm_squared(weight_list):
    """
    Computes a loss that encourages each weight in the list of weights to have unit l2 norm.
    Args:
        weight_list: List of Tensorflow Variables
    Returns:
        w_norm_loss: 0.5 * sum of (1 - l2_norm(w))^2 for each w in weight_list
    """
    w_norm_list = []
    for w in weight_list:
        reduc_dim = list(range(1, w.shape))
        w_norm = tf.reduce_sum(tf.square(1 - tf.sqrt(tf.reduce_sum(tf.square(w), axis=reduc_dim))))
        w_norm_list.append(w_norm)
    norm_loss = 0.5 * tf.add_n(w_norm_list)
    return norm_loss


def weight_decay(weight_list):
    """
    computes typical weight decay loss
    Args:
        weight_list: List of Tensorflow Variables
    Returns:
        decay_loss: 0.5 * sum of w^2 for each w in weight_list
    """
    decay_loss = 0.5 * tf.add_n([tf.reduce_sum(tf.square(w)) for w in weight_list])
    return decay_loss
