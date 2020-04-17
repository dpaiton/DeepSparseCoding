import torch


def reshape_data(data, flatten=None, out_shape=None):
    """
    Helper function to reshape input data for processing and return data shape
    Inputs:
        data: [tensor] data of shape:
            n is num_examples, i is num_rows, j is num_cols, k is num_channels, l is num_examples = i*j*k
            if out_shape is not specified, it is assumed that i == j
            (l) - single data point of shape l, assumes 1 color channel
            (n, l) - n data points, each of shape l (flattened)
            (i, j, k) - single datapoint of of shape (i,j, k)
            (n, i, j, k) - n data points, each of shape (i,j,k)
        flatten: [bool or None] specify the shape of the output
            If out_shape is not None, this arg has no effect
            If None, do not reshape data, but add num_examples dimension if necessary
            If True, return ravelled data of shape (num_examples, num_elements)
            If False, return unravelled data of shape (num_examples, sqrt(l), sqrt(l), 1)
                where l is the number of elements (dimensionality) of the datapoints
            If data is flat and flatten==True, or !flat and flatten==False, then None condition will apply
        out_shape: [list or tuple] containing the desired output shape
            This will overwrite flatten, and return the input reshaped according to out_shape
    Outputs:
        tuple containing:
        data: [tensor] data with new shape
            (num_examples, num_rows, num_cols, num_channels) if flatten==False
            (num_examples, num_elements) if flatten==True
        orig_shape: [tuple of int32] original shape of the input data
        num_examples: [int32] number of data examples or None if out_shape is specified
        num_rows: [int32] number of data rows or None if out_shape is specified
        num_cols: [int32] number of data cols or None if out_shape is specified
        num_channels: [int32] number of data channels or None if out_shape is specified
    """
    orig_shape = data.shape
    orig_ndim = data.dim()
    if out_shape is None:
        if orig_ndim == 1: # single datapoint
            num_examples = 1
            num_channels = 1
            num_elements = orig_shape[0]
            if flatten is None:
                num_rows = num_elements
                num_cols = 1
                data = torch.reshape(data, [num_examples]+list(orig_shape)) # add num_examples=1 dimension
            elif flatten == True:
                num_rows = num_elements
                num_cols = 1
                data = torch.reshape(data, (num_examples, num_rows*num_cols*num_channels))
            else: # flatten == False
                sqrt_num_elements = torch.sqrt(num_elements)
                assert torch.floor(sqrt_num_elements) == torch.ceil(sqrt_num_elements), (
                    "Data length must have an even square root. Note that num_channels is assumed to be 1."
                    +" data length = "+str(num_elements)
                    +" and data_shape="+str(orig_shape))
                num_rows = int(sqrt_num_elements)
                num_cols = num_rows
                data = torch.reshape(data, (num_examples, num_rows, num_cols, num_channels))
        elif orig_ndim == 2: # already flattened
            (num_examples, num_elements) = data.shape
            if flatten is None or flatten == True: # don't reshape data
                num_rows = num_elements
                num_cols = 1
                num_channels = 1
            elif flatten == False:
                sqrt_num_elements = torch.sqrt(num_elements)
                assert torch.floor(sqrt_num_elements) == torch.ceil(sqrt_num_elements), (
                    "Data length must have an even square root when not specifying out_shape.")
                num_rows = int(sqrt_num_elements)
                num_cols = num_rows
                num_channels = 1
                data = torch.reshape(data, (num_examples, num_rows, num_cols, num_channels))
            else:
                assert False, ("flatten argument must be True, False, or None")
        elif orig_ndim == 3: # single data point
            num_examples = 1
            num_rows, num_cols, num_channels = data.shape
            if flatten == True:
                data = torch.reshape(data, (num_examples, num_rows * num_cols * num_channels))
            elif flatten is None or flatten == False: # already not flat
                data = data[None, ...]
            else:
                assert False, ("flatten argument must be True, False, or None")
        elif orig_ndim == 4: # not flat
            num_examples, num_rows, num_cols, num_channels = data.shape
            if flatten == True:
                data = torch.reshape(data, (num_examples, num_rows*num_cols*num_channels))
        else:
            assert False, ("Data must have 1, 2, 3, or 4 dimensions.")
    else:
        num_examples = None; num_rows=None; num_cols=None; num_channels=None
        data = torch.reshape(data, out_shape)
    return (data, orig_shape, num_examples, num_rows, num_cols, num_channels)


def check_all_same_shape(tensor_list):
    """
    Verify that all tensors in the tensor list have the same shape
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
        resh_map  = torch.reshape(feature_map, [-1, prev_input_features])
    elif(map_shape.ndims == 2):
        resh_map = feature_map
    else:
      raise ValueError("Input feature_map has incorrect ndims")
    return resh_map


def standardize(data, eps=None):
  """
  Standardize each image data to have zero mean and unit standard-deviation (z-score)
  Inputs:
      data: [tensor] unnormalized data
  Outputs:
      data: [tensor] normalized data
  """
  if eps is None:
      eps = 1.0 / torch.sqrt(data[0,...].numel())
  data, orig_shape = reshape_data(data, flatten=True)[:2] # Adds channel dimension if it's missing
  num_examples = data.shape[0]
  data_axis = tuple(range(data.dim())[1:]) # standardize each example individually
  data_mean = torch.mean(data, dim=data_axis, keepdim=True)
  data_true_std = torch.std(data, dim=data_axis, keepdim=True)
  data_std = torch.where(data_true_std >= eps, data_true_std, eps*torch.ones_like(data_true_std))
  for idx in range(data.shape[0]): # TODO: Broadcasting should work here
      data[idx, ...] = (data[idx, ...] - data_mean[idx]) /  data_std[idx]
  if data.shape != orig_shape:
      data = reshape_data(data, out_shape=orig_shape)[0]
  return data, data_mean, data_std


def rescale_data_to_one(data):
  """
  Rescale input data to be between 0 and 1, per example
  Inputs:
    data: [tensor] unnormalized data
  Outputs:
    data: [tensor] centered data of shape (n, i, j, k) or (n, l)
  """
  data, orig_shape = reshape_data(data, flatten=None)[:2]
  data_axis = tuple(range(data.ndim)[1:])
  data_min = torch.min(data, axis=data_axis, keepdims=True)
  data_max = torch.max(data, axis=data_axis, keepdims=True)
  data = (data - data_min) / (data_max - data_min + 1e-8)
  if data.shape != orig_shape:
    data = reshape_data(data, out_shape=orig_shape)[0]
  return data, data_min, data_max
