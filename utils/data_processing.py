import numpy as np
import torch


def reshape_data(data, flatten=None, out_shape=None):
    """
    Reshape input data for processing and return data shape

    Keyword arguments:
        data: [tensor] data of shape:
            n is num_examples, i is num_channels (a.k.a. vector length for fattened inputs), j is num_rows, k is num_cols
            if out_shape is not specified, it is assumed that i == j
            (i) - single data point of shape i (flattened; assumed j = k = 1)
            (n, i) - n data points, each of shape i (flattened; assumed j = k = 1)
            (i, j, k) - single datapoint of of shape (i, j, k)
            (n, i, j, k) - n data points, each of shape (i, j, k)
        flatten: [bool or None] specify the shape of the output
            If out_shape is not None, this arg has no effect
            If None, do not reshape data, but add num_examples dimension if necessary
            If True, return ravelled data of shape (num_examples, num_elements), where num_elements=i*j*k
            If False, return unravelled data of shape (num_examples, 1, sqrt(i), sqrt(i))
                where i is the number of elements (size) of the data points and 1 is the assumed number of channels
            If data is flat and flatten==True, or !flat and flatten==False, then None condition will apply
        out_shape: [list or tuple] containing the desired output shape
            This will overwrite flatten, and return the input reshaped according to out_shape

    Outputs:
        tuple containing:
        data: [tensor] data with new shape
            (num_examples, num_channels, num_rows, num_cols) if flatten==False
            (num_examples, num_elements) if flatten==True, where num_elements = num_channels*num_rows*num_cols
        orig_shape: [tuple of int32] original shape of the input data
        num_examples: [int32] number of data examples or None if out_shape is specified
        num_channels: [int32] number of data channels or None if out_shape is specified
        num_rows: [int32] number of data rows or None if out_shape is specified
        num_cols: [int32] number of data cols or None if out_shape is specified
    """
    orig_shape = data.shape
    orig_ndim = data.ndim
    if out_shape is None:
        if orig_ndim == 1: # single datapoint
            num_examples = 1
            num_channels = 1
            num_elements = orig_shape[0]
            if flatten is None:
                num_rows = num_elements
                num_cols = None
                num_channels = None
                data = torch.reshape(data, [num_examples]+list(orig_shape)) # add num_examples=1 dimension
            elif flatten == True:
                num_rows = num_elements
                num_cols = 1
                data = torch.reshape(data, (num_examples, num_channels * num_rows * num_cols))
            else: # flatten == False
                sqrt_num_elements = np.sqrt(num_elements)
                assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
                    'Data length must have an even square root. Note that num_channels is assumed to be 1.'
                    +' data length = '+str(num_elements)
                    +' and data_shape='+str(orig_shape))
                num_rows = int(sqrt_num_elements)
                num_cols = num_rows
                data = torch.reshape(data, (num_examples, num_channels, num_rows, num_cols))
        elif orig_ndim == 2: # already flattened
            (num_examples, num_elements) = data.shape
            if flatten is None or flatten == True: # don't reshape data
                num_rows = num_elements
                num_cols = 1
                num_channels = 1
            elif flatten == False:
                sqrt_num_elements = np.sqrt(num_elements)
                assert np.floor(sqrt_num_elements) == np.ceil(sqrt_num_elements), (
                    'Data length must have an even square root when not specifying out_shape.')
                num_rows = int(sqrt_num_elements)
                num_cols = num_rows
                num_channels = 1
                data = torch.reshape(data, (num_examples, num_channels, num_rows, num_cols))
            else:
                assert False, ('flatten argument must be True, False, or None')
        elif orig_ndim == 3: # single data point
            num_examples = 1
            num_channels, num_rows, num_cols = data.shape
            if flatten == True:
                data = torch.reshape(data, (num_examples, num_channels * num_rows * num_cols))
            elif flatten is None or flatten == False: # already not flat
                data = data[None, ...] # add singleton num_examples dimension
            else:
                assert False, ('flatten argument must be True, False, or None')
        elif orig_ndim == 4: # multiple data points, not flat
            num_examples, num_channels, num_rows, num_cols = data.shape
            if flatten == True:
                data = torch.reshape(data, (num_examples, num_channels * num_rows * num_cols))
        else:
            assert False, ('Data must have 1, 2, 3, or 4 dimensions.')
    else:
        num_examples = None; num_rows=None; num_cols=None; num_channels=None
        data = torch.reshape(data, out_shape)
    return (data, orig_shape, num_examples, num_channels, num_rows, num_cols)


def check_all_same_shape(tensor_list):
    """
    Verify that all tensors in the tensor list have the same shape

    Keyword arguments:
        tensor_list: list of tensors to be checked
    Returns:
        raises error if the tensors are not the same shape
    """
    first_shape = tensor_list[0].shape
    for index, tensor in enumerate(tensor_list):
        if tensor.shape != first_shape:
            raise ValueError(
                f'Tensor entry {index} in input list has shape {tensor.shape}, but should have shape {first_shape}')


def flatten_feature_map(feature_map):
    """
    Flatten input tensor from [batch, c, y, x] to [batch, c * y * x]

    Keyword arguments:
        feature_map: tensor with shape [batch, c, y, x]

    Returns:
        reshaped_map: tensor with  shape [batch, c * y * x]
    """
    map_shape = feature_map.shape
    if(len(map_shape) == 4):
        (batch, c, y, x) = map_shape
        prev_input_features = int(c * y * x)
        resh_map  = torch.reshape(feature_map, [batch, prev_input_features])
    elif(len(map_shape) == 2):
        resh_map = feature_map
    else:
        raise ValueError('Input feature_map has incorrect ndims')
    return resh_map


def get_std_from_dataloader(loader, dataset_mean):
    """
    TODO: Calculate the standard deviation from all entries in a pytorch data loader

    Keyword arguments:
        loader: [pytorch DataLoader] containing the full dataset.
            This function assumes there is always a target label, i.e. loader.next() returns (data, target)
        dataset_mean: [torch tensor] of the same shape as a single dataset sample

    Outputs:
        dataset_std: [torch tensor] of the same shape as a single dataset sample
    """
    #dataset_std = torch.zeros(next(iter(loader)).shape[1:])
    #for data, target in loader:
    #    std_sum_squares += (data - dataset_mean)**2
    #dataset_std = torch.sqrt(std_sum_squares / len(loader.dataset))
    #return dataset_std
    raise NotImplementedError


def get_mean_from_dataloader(loader):
    """
    Calculate the mean datapoint from all entries in a pytorch data loader

    Keyword arguments:
        loader: [pytorch DataLoader] containing the full dataset.
            This function assumes there is always a target label, i.e. loader.next() returns (data, target)

    Outputs:
        dataset_mean: [torch tensor] of the same shape as a single dataset sample
    """
    dataset_mean = torch.zeros(next(iter(loader))[0].shape[1:]) # don't include batch dimension
    num_batches = 0
    for data, target in loader:
        dataset_mean += data.mean(dim=0, keepdim=False)
        num_batches += 1
    return dataset_mean / num_batches


def center(data, samplewise=False, batch_size=100):
    """
    Center image dataset to have zero mean
    
    Keyword arguments:
        data: [tensor] unnormalized data
        samplewise: [bool] if True, center each sample individually; if False, compute mean over entire batch

    Outputs:
        data: [tensor] centered data
    """
    data, orig_shape = reshape_data(data, flatten=True)[:2]
    if(samplewise): # center each input sample individually
        data_axis = tuple(range(data.ndim)[1:])
        data_mean = torch.mean(data, dim=data_axis, keepdim=True)
    else: # center the entire population
        data_mean = torch.mean(data, dim=0)
    data = data - data_mean
    if(data.shape != orig_shape):
        data = reshape_data(data, out_shape=orig_shape)[0]
    return data, data_mean


def standardize(data, eps=None, samplewise=False, batch_size=100):
    """
    Standardize each image data to have zero mean and unit standard-deviation (z-score)
    
    This function uses population standard deviation data.sum() / N, where N = data.shape[0].

    Keyword arguments:
        data: [tensor] unnormalized data
        eps: [float] if the std(data) is less than eps, then divide by eps instead of std(data)
            defaults to 1/sqrt(data_dim) where data_dim is the total size of a data vector
        samplewise: [bool] if True, standardize each sample individually; akin to contrast-normalization
            if False, compute mean and std over entire batch

    Outputs:
        data: [tensor] normalized data
    """
    if(eps is None):
        eps = 1.0 / data[0,...].numel()
    data, orig_shape = reshape_data(data, flatten=True)[:2]
    num_examples = data.shape[0]
    if(samplewise): # standardize each input sample individually
        data_axis = tuple(range(data.ndim)[1:])
        data_mean = torch.mean(data, dim=data_axis, keepdim=True)
        data_true_std = torch.std(data, unbiased=False, dim=data_axis, keepdim=True)
    else: # standardize the entire population
        data_mean = torch.mean(data, dim=0, keepdim=True)
        data_true_std = torch.std(data, dim=0, unbiased=False, keepdim=True)
    data_std = torch.where(data_true_std >= eps, data_true_std, eps*torch.ones_like(data_true_std))
    data = (data - data_mean) /  data_std
    if(data.shape != orig_shape):
        data = reshape_data(data, out_shape=orig_shape)[0]
    return data, data_mean, data_std


def rescale_data_to_one(data, eps=None, samplewise=True):
    """
    Rescale input data to be between 0 and 1

    Keyword arguments:
        data: [tensor] unnormalized data
        eps: [float] if the std(data) is less than eps, then divide by eps instead of std(data)
        samplewise: [bool] if True, compute it per-sample, otherwise normalize entire batch

    Outputs:
        data: [tensor] centered data of shape (n, i, j, k) or (n, l)
    """
    if(eps is None):
        eps = 1.0 / np.sqrt(data[0,...].numel())
    if(samplewise):
        data_min = torch.min(data.view(-1, np.prod(data.shape[1:])),
                             axis=1, keepdim=False)[0].view(-1, *[1]*(data.ndim-1))
        data_max = torch.max(data.view(-1, np.prod(data.shape[1:])),
                             axis=1, keepdim=False)[0].view(-1, *[1]*(data.ndim-1))
    else:
        data_min = torch.min(data)
        data_max = torch.max(data)
    true_range = data_max - data_min
    data_range = torch.where(true_range >= eps, true_range, eps*torch.ones_like(true_range))
    data = (data - data_min) / data_range
    return data, data_min, data_max


def one_hot_to_dense(one_hot_labels):
    """
    Convert a matrix of one-hot labels to a list of dense labels

    Keyword arguments:
        one_hot_labels: one-hot torch tensor of shape [num_labels, num_classes]

    Outputs:
        dense_labels: 1D torch tensor array of labels
            The integer value indicates the class and 0 is assumed to be a class.
            The integer class also indicates the index for the corresponding one-hot representation
    """
    num_labels, num_classes = one_hot_labels.shape
    dense_labels = torch.zeros(num_labels)
    for label_id in range(num_labels):
        dense_labels[label_id] = torch.nonzero(one_hot_labels[label_id, :] == 1)
    return dense_labels


def dense_to_one_hot(labels_dense, num_classes):
    """
    Converts a (np.ndarray) vector of dense labels to a (np.ndarray) matrix of one-hot labels. E.g. [0, 1, 1, 3] -> [00, 01, 01, 11]
    
    Keyword arguments:
        labels_dense: dense torch tensor of shape [num_classes], where each entry is an integer indicating the class label
        num-classes: The total number of classes in the dataset

    Outputs:
        one_hot_labels: one-hot torch tensor of shape [num_labels, num_classes]
    """
    num_labels = labels_dense.shape[0]
    index_offset = torch.arange(end=num_labels, dtype=torch.int32) * num_classes
    labels_one_hot = torch.zeros((num_labels, num_classes))
    labels_one_hot.view(-1)[index_offset + labels_dense.view(-1)] = 1
    return labels_one_hot


def atleast_kd(x, k):
    """
    Return x reshaped to append singleton dimensions such that x.ndim is at least k

    Keyword arguments:
        x [Tensor or numpy ndarray]
        k [int] minimum number of dimensions

    Outputs:
        x [same as input x] reshaped input to have at least k dimensions
    """
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)


def get_weights_l2_norm(w, eps=1e-12):
    """
    Return l2 norm of weight matrix

    Keyword arguments:
        w [Tensor] assumed to have shape [inC, outC] or [outC, inC, kernH, kernW]
            norm is calculated over vectorized version of inC in the first case or inC*kernH*kernW in the second
        eps [float] minimum value to prevent division by zero

    Outputs:
        norm [Tensor] norm of each of the outC weight vectors
    """
    if w.ndim == 2: # fully-connected, [inputs, outputs]
        norms = torch.norm(w, dim=0, keepdim=True)
    elif w.ndim == 4: # convolutional, [out_channels, in_channels, kernel_height, kernel_width]
        norms = torch.norm(w.flatten(start_dim=1), dim=-1, keepdim=True)
    else:
        assert False, (f'input w must have ndim = 2 or 4, not {w.ndim}')
    if(torch.max(norms) <= eps): #TODO: Warnings
        print(f'Warning: input gradient is less than or equal to {eps}')
    norms = torch.max(norms, eps*torch.ones_like(norms)) # prevent div by 0 # TODO: Change to torch.maximum when it is stable
    norms = atleast_kd(norms, w.ndim)
    return norms


def l2_normalize_weights(w, eps=1e-12):
    """
    l2 normalize weight matrix

    Keyword arguments:
        w [Tensor] assumed to have shape [inC, outC] or [outC, inC, kernH, kernW]
            norm is calculated over vectorized version of inC in the first case or inC*kernH*kernW in the second
        eps [float] minimum value to prevent division by zero

    Outputs:
        w [Tensor] same type and shape as input w, but with unitary l2 norm when computed over all input dimensions
    """
    norms = get_weights_l2_norm(w, eps)
    return w / norms


def single_image_to_patches(image, patch_shape):
    """
    Extract patches from a single image

    Keyword arguments:
        image [torch tensor] of shape [im_chan, im_height, im_width]
        patch_shape [tuple or list] containing the output shape
            [patch_chan, patch_height, patch_width]
            patch_chan must be the same as im_chan

        It is recommended, though not required, that the patch height and width divide evenly into
        the image height and width, respectively.

    Outputs:
        patches [torch tensor] of patches of shape [num_patches]+list(patch_shape)
    """
    try:
        im_chan, im_height, im_width = image.shape
        patch_chan, patch_height, patch_width = patch_shape
    except Exception as e:
        raise ValueError(
            f'This function requires that: '
            +f'1) The input variable "image" must have shape [im_chan, im_height, im_width], and is  {image.shape}'
            +f'and 2) the input variable "patch_shape" must have shape [patch_chan, patch_height, patch_width], and is {patch_shape}.'
        ) from e
    num_row_patches = np.floor(im_height / patch_height)
    num_col_patches = np.floor(im_width / patch_width)
    num_patches = int(num_row_patches * num_col_patches)
    patches = torch.zeros((num_patches, patch_chan, patch_height, patch_width))
    row_id = 0
    col_id = 0
    for patch_idx in range(num_patches):
        row_end = row_id + patch_height
        col_end = col_id + patch_width
        try:
            patches[patch_idx, ...] = image[:, row_id:row_end, col_id:col_end]
        except Exception as e:
            raise ValueError('This function requires that im_chan equal patch_chan.') from e
        row_id += patch_height
        if row_id >= im_height:
            row_id = 0
            col_id += patch_width
        if col_id >= im_width:
            col_id = 0
    return patches


def patches_to_single_image(patches, image_shape):
    """
    Convert patches input into a single ouput

    Keyword arguments:
          patches [torch tensor] of shape [num_patches, patch_chan, patch_height, patch_width]
          image_shape [list or tuple] of length 2 containing the image shape [im_chan, im_height, im_width]

        im_chan is assumed to equal patch_chan

    Outputs:
        image [torch tensor] of shape [im_chan, im_height, im_width]
    """
    try:
        num_patches, patch_chan, patch_height, patch_width = patches.shape
        im_chan, im_height, im_width = image_shape
    except Exception as e:
        raise ValueError(
            f'This funciton requires that input patches has shape'
            f' [num_patches, patch_chan, patch_height, patch_width] and is {patches.shape}'
            f' and input image_shape is a list or tuple of integers of length 3 containing [im_chan, im_height, im_width] and is {image_shape}'
        ) from e
    image = torch.zeros((im_chan, im_height, im_width))
    row_id = 0
    col_id = 0
    for patch_idx in range(num_patches):
        row_end = row_id + patch_height
        col_end = col_id + patch_width
        image[:, row_id:row_end, col_id:col_end] = patches[patch_idx, ...]
        row_id += patch_height
        if row_id >= im_height:
            row_id = 0
            col_id += patch_width
        if col_id >= im_width:
            col_id = 0
    return image

def images_to_patches(images, patch_shape):
    """
    Extract evenly distributed non-overlapping patches from an image dataset

    Keyword arguments:
        images [torch tensor] of shape [num_images, im_chan, im_height, im_width] or [im_chan, im_height, im_width] for a single image
        patch_shape [tuple or list] containing the output shape
            [patch_chan, patch_height, patch_width]
            patch_chan must be the same as im_chan

        It is recommended, though not required, that the patch height and width divide evenly into the image height and width, respectively.

    Outputs:
        patches [np.ndarray] of patches of shape [num_patches]+list(patch_shape)
    """
    if images.ndim == 3: # single image
        return single_image_to_patches(images, patch_shape)
    num_im, im_chan, im_height, im_width = images.shape
    patch_chan, patch_height, patch_width = patch_shape
    num_row_patches = np.floor(im_height / patch_height)
    num_col_patches = np.floor(im_width / patch_width)
    num_patches_per_im = int(num_row_patches * num_col_patches)
    tot_num_patches =  int(num_patches_per_im * num_im)
    patches = torch.zeros([tot_num_patches, ]+list(patch_shape))
    patch_id = 0
    for im_id in range(num_im):
        image = images[im_id, ...]
        image_patches = single_image_to_patches(image, patch_shape)
        patch_end = patch_id + num_patches_per_im
        patches[patch_id:patch_end, ...] = image_patches
        patch_id += num_patches_per_im
    return patches

def patches_to_images(patches, image_shape):
    """
    Recombine patches tensor into a dataset of images

    Keyword arguments:
        patches [torch tensor] holding square patch data of shape [num_patches, patch_chan, patch_height, patch_width]
        image_shape [list or tuple] containing the image dataset shape [im_chan, im_height, im_width]

        It is assumed that im_chan equals patch_chan

    Outputs:
        images [torch tensor] holding the recombined image dataset
    """
    tot_num_patches, patch_chan, patch_height, patch_width = patches.shape
    im_chan, im_height, im_width = image_shape
    num_row_patches = np.floor(im_height / patch_height)
    num_col_patches = np.floor(im_width / patch_width)
    num_patches_per_im = int(num_row_patches * num_col_patches)
    num_im = tot_num_patches // num_patches_per_im
    images = torch.zeros([num_im]+image_shape)
    patch_id = 0
    for im_id in range(num_im):
        patch_end = patch_id + num_patches_per_im
        patch_batch = patches[patch_id:patch_end, ...]
        images[im_id, ...] = patches_to_single_image(patch_batch, image_shape)
        patch_id += num_patches_per_im
    return images


def covariance(tensor):
    """
    Returns the covariance matrix of the input tensor

    Keyword arguments:
        tensor [torch tensor] of shape [num_batch, num_channels] or [num_batch, num_channels, elements_h, elements_w]
        if tensor.ndim is 2 then the covariance is computed for each element over the batch dimension
        if tensor.ndim is 4 then the covariance is computed over spatial dimensions for each channel and each batch instance

    Outputs:
        covariance matrix [torch tensor] of shape [num_channels, num_channels]
    """
    if tensor.ndim == 2: # [num_batch, num_channels]
        centered_tensor = tensor - tensor.mean(dim=0, keepdim=True) # subtract mean vector
        corvariance = torch.dot(centered_tensor.T, centered_tensor) # sum over batch
    elif tensor.ndim == 4: # [num_batch, num_channels, elements_h, elements_w]
        num_batch, num_channels, elements_h, elements_w = tensor.shape
        flat_map = tensor.view(num_batch, num_channels, elements_h * elements_w)
        cent_flat_map = flat_map - flat_map.mean(dim=2, keepdim=True) # subtract mean vector
        covariance = torch.bmm(cent_flat_map, torch.transpose(cent_flat_map, 1, 2)) # sum over space
        covariance = covariance.mean(dim=0, keepdim=False) # avg cov over batch
    return covariance
