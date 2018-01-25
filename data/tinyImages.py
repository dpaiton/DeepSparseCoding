import matplotlib
matplotlib.use("Agg")

import numpy as np
from data.dataset import Dataset
import utils.plot_functions as pf
import time as ti

class tinyImages(object):
  def __init__(self, image_dir, num_batches=1, rand_state=np.random.RandomState()):
    self.total_num_images = 79302017
    self.image_shape = [32, 32, 3]
    self.num_pixels = int(np.prod(self.image_shape))
    self.images = self.extract_random_images(image_dir, num_batches, rand_state)

  def extract_random_images(self, image_dir, num_batches=None, rand_state=np.random.RandomState()):
    batch_size = 60000 # CIFAR sized batches
    self.indices = rand_state.permutation(self.total_num_images)
    if num_batches is None:
        num_batches = self.total_num_images/batch_size
    file_location = image_dir+"tiny_images.bin"
    with open(file_location, "rb") as data_file:
      images = []
      for img_index in range(0, num_batches*batch_size, batch_size):
        offset = img_index * self.num_pixels
        data_file.seek(offset)
        data = np.fromstring(data_file.read(self.num_pixels*batch_size), dtype='uint8')
        data = np.stack(np.split(data, batch_size))
        images.append(data.reshape([batch_size]+self.image_shape, order="F"))
    return np.concatenate(images, axis=0)

def load_tinyImages(kwargs):
  data_dir = kwargs["data_dir"]
  rand_state = kwargs["rand_state"] if "rand_state" in kwargs.keys() else np.random.RandomState()
  num_batches = int(kwargs["num_batches"]) if "num_batches" in kwargs.keys() else None
  ti_data = tinyImages(data_dir, num_batches, rand_state)
  pf.plot_data_tiled(ti_data.images[1000:1100,...], cmap=None, save_filename="./tmp.png")
  image_dataset = Dataset(ti_data.images, lbls=None, ignore_lbls=None, rand_state=rand_state)
  return {"train":image_dataset}
