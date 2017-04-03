import numpy as np
import utils.image_processing as ip

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls=None, vectorize=True,
    rand_state=np.random.RandomState()):
    if imgs.ndim == 3:
      (self.num_examples, self.num_rows, self.num_cols) = imgs.shape
      self.num_channels = 1
    elif imgs.ndim == 4:
      (self.num_examples, self.num_rows,
        self.num_cols, self.num_channels) = imgs.shape
    else:
      assert False, (
        "ndim must be 3 (batch, rows, cols) or 4 (batch, rows, cols, chans)")
    if vectorize:
      self.images = imgs.reshape(self.num_examples,
        self.num_rows*self.num_cols*self.num_channels)
    else:
      self.images = imgs
    self.num_pixels = self.num_rows*self.num_cols*self.num_channels
    self.labels = lbls
    self.ignore_labels = ignore_lbls
    self.epochs_completed = 0
    self.batches_completed = 0
    self.curr_epoch_idx = 0
    self.rand_state = rand_state
    self.epoch_order = self.rand_state.permutation(self.num_examples)

  """
  Advance epoch counter & generate new index order
  Inputs:
    num_to_advance [int] number of epochs to advance
  """
  def new_epoch(self, num_to_advance=1):
    self.epochs_completed += int(num_to_advance)
    for _ in range(int(num_to_advance)):
      self.epoch_order = self.rand_state.permutation(self.num_examples)

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
  def next_batch(self, batch_size):
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

  """
  Increment member variables to reflect a step forward of num_batches images
  Inputs:
    num_batches: How many batches to step forward
    batch_size: How many examples constitute a batch
  """
  def advance_counters(self, num_batches, batch_size):
    assert self.curr_epoch_idx == 0, ("Error: Current epoch index must be 0.")
    if num_batches * batch_size > self.num_examples:
      self.new_epoch(int((num_batches * batch_size) / float(self.num_examples)))
    self.batches_completed += num_batches
    self.curr_epoch_idx = (num_batches * batch_size) % self.num_examples

  """Reshape images to be a vector per data point"""
  def vectorize_data(self):
    #assert self.images.ndim == 4, ("Image must be a 4D tensor")
    self.images = self.images.reshape(self.num_examples,
      self.num_rows * self.num_cols * self.num_channels)

  """Reshape images to be a vector per data point"""
  def devectorize_data(self):
    #assert self.images.ndim == 2, ("Image must be a 2D tensor")
    self.images = self.images.reshape(self.num_examples,
      self.num_rows, self.num_cols, self.num_channels)
