import numpy as np
import utils.data_processing as dp

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls=None, vectorize=True,
    rand_state=np.random.RandomState()):
    self.vectorize = vectorize
    if imgs.ndim == 3:
      (self.num_examples, self.num_rows, self.num_cols) = imgs.shape
      #imgs = imgs[..., None] # Need to keep singleton dimension for color
      self.num_channels = 1
    elif imgs.ndim == 4:
      (self.num_examples, self.num_rows,
        self.num_cols, self.num_channels) = imgs.shape
    else:
      assert False, ("data ndim must be 3 (batch, rows, cols) or 4 (batch, rows, cols, chans)")
    if self.vectorize:
      self.images = dp.reshape_data(imgs, flatten=True)
    else:
      self.images = imgs
    self.ndim = imgs.ndim
    self.shape = imgs.shape
    self.num_pixels = np.prod(self.shape[1:])
    self.labels = lbls
    self.ignore_labels = ignore_lbls
    self.rand_state = rand_state
    self.reset_counters()

  def reset_counters(self):
    """
      Reset all counters for batches & epochs completed
    """
    self.epochs_completed = 0
    self.batches_completed = 0
    self.curr_epoch_idx = 0
    self.epoch_order = self.rand_state.permutation(self.num_examples)

  def downsample(self, scale_factor=None, order=3):
    """
    Downsample data with scipy.ndimage.interpolation.zoom
    Inputs:
      data: np.ndarray
      scale_factor [list of floats] indicating the downsampling factor for each dimension
        Values in the list should be between 0.0 and 1.0
        scale_factor needs an element for each dimension in the data
      order: [int 0-5] the order for the spline interpolation
    """
    if scale_factor is None:
      scale_factor = [1.0,]*data.ndim
    else:
      assert len(scale_factor) == self.images.ndim, ("len(scale_factor) must == data.ndim")
    self.images = dp.downsample_data(self.images, scale_factor=scale_factor, order=order)
    self.shape = self.images.shape
    (self.num_rows, self.num_cols) = self.shape[1:]
    #(self.num_rows, self.num_cols, self.num_channels) = self.shape[1:]
    self.num_pixels = np.prod(self.shape[1:])

  def preprocess(self, params):
    """
    Perform default preprocessing on the self.images object
    Possible kwargs are:
      center_data: subtract mean from data
      norm_data: divide data by the maximum
      whiten_data: default method is using the Fourier amplitude spectrium ("FT")
        change default with whiten_method param
      standardize_data: subtract mean and divide by the standard deviation
      contrast_normalize: divide by gaussian blurred surround pixels
      extract_patches: break up data into patches
        see utils/data_processing/exract_patches() for docs
    """
    if "center_data" in params.keys():
      if params["center_data"]:
        self.images = dp.center_data(self.images, use_dataset_mean=False)
    if "norm_data" in params.keys():
      if params["norm_data"]:
        self.images = dp.normalize_data_with_max(self.images)
    if "whiten_data" in params.keys(): # if FT method, whiten before patching
      if params["whiten_data"]:
        if "whiten_method" in params.keys():
          if params["whiten_method"] == "FT":
            self.images, self.w_filter = dp.whiten_data(self.images,
              method=params["whiten_method"])
        else:
          self.images, self.w_filter = dp.whiten_data(self.images, method="FT")
    if "standardize_data" in params.keys():
      if params["standardize_data"]:
        self.images = dp.standardize_data(self.images)
    if "contrast_normalize" in params.keys():
      if params["contrast_normalize"]:
        if "gauss_patch_size" in params.keys():
          self.images = dp.contrast_normalize(self.images, params["gauss_patch_size"])
        else:
          self.images = dp.contrast_normalize(self.images)
    if "extract_patches" in params.keys():
      if params["extract_patches"]:
        assert all(key in params.keys()
          for key in ["num_patches", "patch_edge_size", "overlapping_patches",
          "randomize_patches"]), ("Insufficient params for patches.")
        out_shape = (int(params["num_patches"]), int(params["patch_edge_size"]**2))
        self.num_examples = np.int32(params["num_patches"])
        self.num_rows = params["patch_edge_size"]
        self.num_cols = params["patch_edge_size"]
        self.num_pixels = np.int32(self.num_rows*self.num_cols*self.num_channels)
        self.reset_counters()
        if "patch_variance_threshold" in params.keys():
          self.images = dp.extract_patches(self.images, out_shape, params["overlapping_patches"],
            params["randomize_patches"], params["patch_variance_threshold"], self.rand_state)
        else:
          self.images = dp.extract_patches(self.images, out_shape, params["overlapping_patches"],
            params["randomize_patches"], var_thresh=0, rand_state=self.rand_state)
    if "whiten_data" in params.keys(): # other whiten methods happen after patching
      if params["whiten_data"]:
        if "whiten_method" in params.keys():
          self.images, self.w_filter = dp.whiten_data(self.images, method=params["whiten_method"])
        self.images, self.w_filter = dp.whiten_data(self.images)

  def new_epoch(self, num_to_advance=1):
    """
    Advance epoch counter & generate new index order
    Inputs:
      num_to_advance [int] number of epochs to advance
    """
    self.epochs_completed += int(num_to_advance)
    for _ in range(int(num_to_advance)):
      self.epoch_order = self.rand_state.permutation(self.num_examples)

  def next_batch(self, batch_size):
    """
    Return a batch of images
    Outputs:
      3d tuple containing images, labels and ignore labels
    Inputs:
      batch_size [int] representing the number of images in the batch
        NOTE: If batch_size does not divide evenly into self.num_examples then
        some of the images will not be delivered. The function assumes that
        batch_size is a scalar increment of num_examples.
    """
    assert batch_size <= self.num_examples, (
        "Input batch_size was greater than the number of available examples.")
    if self.curr_epoch_idx + batch_size > self.num_examples:
      start = 0
      self.new_epoch(1)
      self.curr_epoch_idx = 0
    else:
      start = self.curr_epoch_idx
    self.batches_completed += 1
    self.curr_epoch_idx += batch_size
    set_indices = self.epoch_order[start:self.curr_epoch_idx]
    if self.labels is not None:
      if self.ignore_labels is not None:
        return (self.images[set_indices, ...],
          self.labels[set_indices, ...],
          self.ignore_labels[set_indices, ...])
      return (self.images[set_indices, ...],
        self.labels[set_indices, ...],
        self.ignore_labels)
    return (self.images[set_indices, ...], self.labels, self.ignore_labels)

  def advance_counters(self, num_batches, batch_size):
    """
    Increment member variables to reflect a step forward of num_batches images
    Inputs:
      num_batches: How many batches to step forward
      batch_size: How many examples constitute a batch
    """
    assert self.curr_epoch_idx == 0, ("Error: Current epoch index must be 0.")
    if num_batches * batch_size > self.num_examples:
      self.new_epoch(int((num_batches * batch_size) / float(self.num_examples)))
    self.batches_completed += num_batches
    self.curr_epoch_idx = (num_batches * batch_size) % self.num_examples
