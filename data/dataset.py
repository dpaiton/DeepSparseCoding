import numpy as np
import utils.data_processing as dp

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls=None, vectorize=True,
    rand_state=np.random.RandomState()):
    self.vectorize = vectorize
    if imgs.ndim == 3:
      (self.num_examples, self.num_rows, self.num_cols) = imgs.shape
      self.num_channels = 1
    elif imgs.ndim == 4:
      (self.num_examples, self.num_rows,
        self.num_cols, self.num_channels) = imgs.shape
    else:
      assert False, ("ndim must be 3 (batch, rows, cols) or 4 (batch, rows, cols, chans)")
    if self.vectorize:
      self.images = imgs.reshape(self.num_examples, self.num_rows*self.num_cols*self.num_channels)
    else:
      self.images = imgs
    self.num_pixels = self.num_rows*self.num_cols*self.num_channels
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

  def preprocess(self, params):
    """
    Perform preprocessing on the self.images object
    Possible kwargs are:
      whiten_data: default method is using the Fourier amplitude spectrium ("FT")
        change default method with params["whiten_method"]
      standardize_data: subtract mean and divide by the standard deviation
      contrast_normalize: divide by gaussian blurred surround pixels
      patches: break up data into patches
        see utils/data_processing/exract_patches() for docs
    """
    if "whiten_data" in params.keys():
      if params["whiten_data"]:
        if "whiten_method" in params.keys():
          self.images = dp.whiten_data(self.images, method=params["whiten_method"])
        else:
          self.images = dp.whiten_data(self.images)
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
        out_shape = (params["num_patches"], int(params["patch_edge_size"]**2))
        self.num_examples = np.int32(params["num_patches"])
        self.num_rows = params["patch_edge_size"]
        self.num_cols = params["patch_edge_size"]
        self.num_pixels = np.int32(self.num_rows*self.num_cols*self.num_channels)
        self.reset_counters()
        if "patch_variance_threshold" in params.keys():
          self.images = dp.extract_patches(self.images, out_shape, params["overlapping_patches"],
            params["randomize_patches"], params["patch_variance_threshold"], self.rand_state)
        else:
          self.images = dp.extract_patches(self.images, out_shape, params["overlapping_patche"],
            params["randomize_patches"], var_thresh=0, rand_state=self.rand_state)

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

  def vectorize_data(self):
    """Reshape images to be a vector per data point"""
    #assert self.images.ndim == 4, ("Image must be a 4D tensor")
    self.images = self.images.reshape(self.num_examples,
      self.num_rows * self.num_cols * self.num_channels)

  def devectorize_data(self):
    """Reshape images to be a vector per data point"""
    #assert self.images.ndim == 2, ("Image must be a 2D tensor")
    self.images = self.images.reshape(self.num_examples,
      self.num_rows, self.num_cols, self.num_channels)
