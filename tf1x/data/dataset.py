import numpy as np

from DeepSparseCoding.tf1x.utils import data_processing as dp

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls=None, rand_state=np.random.RandomState()):
    """
    Inputs:
      imgs [np.ndarray] of ndim 4 (examples, rows, cols, channels)
      lbls [np.ndarray] of ndim 2 (examples, labels)
      ignore_lbls [np.ndarray] of same shape as lbls
      rand_state [np.random.RandomState]
    """
    assert imgs.ndim == 4, ("data ndim must be 4 (batch, rows, cols, chans)")
    (self.num_examples, self.num_rows, self.num_cols, self.num_channels) = imgs.shape
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

  def downsample(self, scale_factor, order=3):
    """
    Downsample data with scipy.ndimage.interpolation.zoom
    Inputs:
      data: np.ndarray
      scale_factor [list of floats] indicating the downsampling factor for each dimension
        Values in the list should be between 0.0 and 1.0
        scale_factor needs an element for each dimension in the data
      order: [int 0-5] the order for the spline interpolation
    """
    assert len(scale_factor) == self.images.ndim, ("len(scale_factor) must == data.ndim")
    self.images = dp.downsample_data(self.images, scale_factor=scale_factor, order=order)
    self.shape = self.images.shape
    (self.num_rows, self.num_cols, self.num_channels) = self.shape[1:]
    self.num_pixels = np.prod(self.shape[1:])

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
        "Input batch_size (%g) was greater than the number of available examples (%g)."%(
        batch_size, self.num_examples))
    if self.curr_epoch_idx + batch_size > self.num_examples:
      start = 0
      self.new_epoch(1)
      self.curr_epoch_idx = 0
    else:
      start = self.curr_epoch_idx
    self.batches_completed += 1
    self.curr_epoch_idx += batch_size
    set_indices = self.epoch_order[start:self.curr_epoch_idx]
    # The following code modifies what is returned to support None type passthrough
    # and also index the relevant numpy arrays
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
