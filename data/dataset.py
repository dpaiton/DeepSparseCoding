import numpy as np

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls, normalize=False,
    rand_state=np.random.RandomState()):
    self.num_examples = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    if normalize:
      self.images = self.normalize_image(imgs.reshape(self.num_examples,
        num_rows*num_cols))
    else:
      self.images = imgs.reshape(self.num_examples, num_rows*num_cols)
      self.images /= 255.0
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
    num_to_advance : [int] number of epochs to advance
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
    batch_size : [int] representing the number of images in the batch
  Function assumes that batch_size is a scalar increment of num_examples.
  """
  def next_batch(self, batch_size):
    if batch_size > self.num_examples:
      batch_size = self.num_examaples
      print("WARNING: batch_size was greater than the number of available"+
        "examples. batch_size has been set to equal the number of available"+
        "examples.")
    if self.num_examples % batch_size != 0:
      print("WARNING: batch_size should divide evenly into"
       +" num_examples. Some images may not be included in the dataset.")
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
        self.labels[set_indices, ...], self.ignore_labels)
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

  """
  Normalize input image to have mean 0 and std 1
  The operation is done per image, not across the batch.
  Outputs:
    norm: normalized image
  Inputs:
    img: numpy ndarray of dim [num_batch, num_data]
  """
  def normalize_image(self, img):
    norm_img = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:])
      for idx
      in range(img.shape[0])])
    return norm_img
