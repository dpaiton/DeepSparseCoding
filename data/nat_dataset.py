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
    if "lpf_data" in params.keys():
      self.lpf_data = params["lpf_data"]
      if "lpf_cutoff" in params.keys():
        self.lpf_cutoff = params["lpf_cutoff"]
      else:
        self.lpf_cutoff = 0.7
    else:
      self.lpf_data = False
    self.batch_means = []
    self.reset_counters()

  def reset_counters(self):
    self.epochs_completed = 0
    self.batches_completed = 0
    self.curr_epoch_idx = 0
    self.batch_means = []
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

  def crop_to_square(self, image):
    """
    crop image to whatever the smallest dimension is
    """
    orig_height, orig_width, orig_channels = image.shape
    if orig_height > orig_width:
      return image[:orig_width, ...]
    elif orig_height < orig_width:
      return image[:, :orig_height, ...]
    return image

  def crop_to_size(self, image, shape):
    orig_shape = image.shape
    if orig_shape[0] > shape[0]:
      image = image[:shape[0], ...]
    if orig_shape[1] > shape[1]:
      image = image[:, :shape[1], ...]
    return image

  def resize_preserving_aspect_then_crop(self, image, new_shape):
    """
    Resize images while preserving the aspect ratio then crop them to desired shape
    Resize is better than crop because it gets rid of possible compression artifacts in training data
    TODO: Crop center out, not from edge
    """
    orig_shape = image.shape
    orig_height = orig_shape[0]
    orig_width = orig_shape[1]
    orig_chans = orig_shape[2]
    if orig_height > orig_width:
      scale = new_shape[0]/orig_width
    else:
      scale = new_shape[1]/orig_height
    new_height = int(orig_height * scale)
    new_width = int(orig_width * scale)
    # resize preserving aspect ratio
    image =  transform.resize(image, [new_height, new_width, orig_chans], anti_aliasing=True,
      mode="reflect")
    # crop to square
    image = self.crop_to_size(image, new_shape)
    # in case original image dim was less than new dim, expand
    image =  transform.resize(image, new_shape, anti_aliasing=True,
      mode="reflect")
    return image

  def preprocess(self, image):
    if image.ndim == 2:
      image = image[:, :, None]
    image = self.resize_preserving_aspect_then_crop(image, self.shape)
    return image

  def load_images(self, indices):
    img_list = []
    for idx in indices:
      image = io.imread(self.filenames[idx], as_gray=True)
      img_list.append(self.preprocess(image))
    try:
      images = np.stack(img_list, axis=0)
    except:
      print("Failed to reformat image list...")
      import IPython; IPython.embed(); raise SystemExit
    if self.lpf_data:
      images, data_mean, lpf_filter = dp.lpf_data(images, cutoff=self.lpf_cutoff)
      self.batch_means.append(data_mean)
    return (images, [self.filenames[idx] for idx in indices])

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
    images, img_locs = self.load_images(set_indices)
    return (images, img_locs)
