import numpy as np
import skimage.transform as transform
import skimage.io as io
import utils.data_processing as dp

class Dataset(object):
  def __init__(self, file_location, params):
    self.filenames = [string.strip() for string in open(file_location, "r").readlines()]
    self.num_examples = len(self.filenames)
    self.rand_state = params["rand_state"]
    self.shape = params["data_shape"]
    self.reset_counters()

  def reset_counters(self):
    self.epochs_completed = 0
    self.batches_completed = 0
    self.curr_epoch_idx = 0
    self.epoch_order = self.rand_state.permutation(self.num_examples)

  def advance_counters(self, num_batches, batch_size):
    if num_batches * batch_size > self.num_examples:
      self.new_epoch(int((num_batches*batch_size) / float(self.num_examples)))
    self.batches_completed += num_batches
    self.curr_epoch_idx = (num_batches * batch_size) % self.num_examples

  def new_epoch(self, num_to_advance=1):
    self.epochs_completed += int(num_to_advance)
    for _ in range(num_to_advance):
      self.epoch_order = self.rand_state.permutation(self.num_examples)

  def crop_and_resize(self, image, new_size):
    # TODO: should crop to square (take half from each side)
    # and then resize down to new_size
    #num_rows, num_cols, num_chans = image.shape
    #new_rows, new_cols, new_chans = new_size
    #if num_rows < new_rows:
    #  if num_cols < new_cols:
    #    img_resh =  transform.resize(image, new_size, anti_aliasing=True)
    #  img_resh = transform.resize(image,
    #    [new_rows, num_cols, num_chans], anti_aliasing=True)[:, :new_cols, ...]
    #img_resh = image[:new_size[0], :new_size[1], ...]
    img_resh = transform.resize(image, new_size, anti_aliasing=True, mode="reflect")
    return img_resh

  def preprocess(self, image):
    # TODO: should probably also LPF
    if image.ndim == 2:
      image = image[:,:,None]
    return self.crop_and_resize(image, self.shape)

  def load_images(self, indices):
    img_list = []
    for idx in indices:
      image = io.imread(self.filenames[idx], as_gray=True)
      img_list.append(self.preprocess(image))
      try:
        img_stack = np.stack(img_list, axis=0)
      except:
        import IPython; IPython.embed(); raise SystemExit
    return img_stack

  def next_batch(self, batch_size):
    if self.curr_epoch_idx + batch_size > self.num_examples:
      start = 0
      self.new_epoch(1)
      self.curr_epoch_idx = 0
    else:
      start = self.curr_epoch_idx
    self.batches_completed += 1
    self.curr_epoch_idx += batch_size
    set_indices = self.epoch_order[start:self.curr_epoch_idx]
    return (self.load_images(set_indices), None)
