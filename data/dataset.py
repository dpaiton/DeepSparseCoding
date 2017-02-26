import numpy as np

class Dataset(object):
  def __init__(self, imgs, lbls, ignore_lbls,
    rand_state=np.random.RandomState()):
    self.num_examples = imgs.shape[0]
    self.num_px_rows = imgs.shape[1]
    self.num_px_cols = imgs.shape[2]
    self.images = imgs.reshape(self.num_examples,
      self.num_px_rows*self.num_px_cols)
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

  """
  Standardize data to have zero mean and  unit variance
    method is in-place and has no inputs or outputs
  """
  def standardize_data(self):
    self.images -= np.mean(self.images)
    self.images = np.vstack([self.images[idx,:] / np.std(self.images[idx,:])
      for idx in range(self.images.shape[0])])

  """
  Whiten data
  Inputs:
    data [np.ndarray] of dim (num_batch, num_data_points)
    method [str] method to use, can be {FT}
  Outputs:
    whitened_data
  """
  def whiten_data(self, method="FT"):
    imgs = self.images[0:10,:]
    assert self.num_px_rows == self.num_px_cols, ("Input should be square")
    if method == "FT":
      #data_shape = (self.num_px_rows, self.num_px_cols, self.num_examples)
      #data = self.images.T.reshape(data_shape) # filter in spatial domain
      data_shape = (self.num_px_rows, self.num_px_cols, 10)
      data = imgs.T.reshape(data_shape) # filter in spatial domain
      dataFT = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1)), axes=(0, 1))
      nyq = self.num_px_rows / 2
      freqs = np.linspace(-nyq, nyq-1, num=self.num_px_rows)
      fspace = np.meshgrid(freqs, freqs)
      rho = np.sqrt(np.square(fspace[0]) + np.square(fspace[1]))
      lpf = np.exp(-0.5 * np.square(rho / (0.7 * nyq)))
      filtf = np.multiply(rho, lpf)
      dataFT_wht = np.multiply(imFT, filtf.reshape((self.num_px_rows,
        self.num_px_rows, 1)))
      data_wht = np.real(np.fft.ifft2(np.fft.ifftshift(imFT_wht, axes=(0, 1)),
        axes=(0, 1)))
      import IPython; IPython.embed(); raise SystemExit
    if method == "PCA":
      #data = self.images - self.images.mean(axis=1)[:, None]
      data = imgs - imgs.mean(axis=1)[:, None]
      Cov = np.cov(data.T) # Covariace matrix
      U, S, V = np.linalg.svd(Cov) # SVD decomposition
      isqrtS = np.diag(1 / np.sqrt(S)) # Inverse sqrt of S
      data_wht = np.dot(np.dot(data, U), isqrtS)
      import IPython; IPython.embed(); raise SystemExit
    return data_wht
