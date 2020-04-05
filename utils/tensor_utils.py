import tensorflow as tf

def check_all_same_shape(tensor_list):
    """
    verify that all tensors in the tensor list have the same shape
    Args:
        tensor_list: list of tensors to be checked
    Returns:
        raises error if the tensors are not the same shape
    """
    first_shape = tensor_list[0].shape
    for index, tensor in enumerate(tensor_list):
        if tensor.shape != first_shape:
            raise ValueError(
                "Tensor entry %g in input list has shape %g, but should have shape %g"%(
                index, tensor.shape, first_shape))


def flatten_feature_map(feature_map):
    """
    Flatten input tensor from [batch, y, x, f] to [batch, y*x*f]
    Args:
        feature_map: tensor with shape [batch, y, x, f]
    Returns:
        reshaped_map: tensor with  shape [batch, y*x*f]
    """
    map_shape = feature_map.get_shape()
    if(map_shape.ndims == 4):
        (batch, y, x, f) = map_shape
        prev_input_features = int(y * x * f)
        resh_map  = tf.reshape(feature_map, [-1, prev_input_features])
    elif(map_shape.ndims == 2):
        resh_map = feature_map
    else:
      raise ValueError("Input feature_map has incorrect ndims")
    return resh_map
