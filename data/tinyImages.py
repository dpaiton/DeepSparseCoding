import numpy as np
from data.dataset import Dataset
import utils.plot_functions as pf

class tinyImages(object):
  def __init__(self, image_dir, num_batches=1, rand_state=np.random.RandomState()):
    self.total_num_images = 79302017
    self.image_shape = [32, 32, 3]
    self.num_pixels = int(np.prod(self.image_shape))
    start_index = 0
    self.images = self.extract_random_images(image_dir, num_batches, start_index, rand_state)

  def extract_random_images(self, image_dir, num_batches=2, start_index=0,
    rand_state=np.random.RandomState()):
    batch_size = 60000 # CIFAR sized batches
    self.indices = rand_state.permutation(self.total_num_images)
    if num_batches is None:
        num_batches = self.total_num_images/batch_size
    file_location = image_dir+"tiny_images.bin"
    with open(file_location, "rb") as data_file:
      images = []
      for img_index in range(start_index, int((num_batches*batch_size)-start_index), batch_size):
        offset = img_index * self.num_pixels
        data_file.seek(offset)
        data = np.fromstring(data_file.read(self.num_pixels*batch_size), dtype='uint8') # TODO: Switch to np.frombuffer()
        data = np.stack(np.split(data, batch_size))
        images.append(data.reshape([batch_size]+self.image_shape, order="F"))
    return np.concatenate(images, axis=0)

def load_tinyImages(params):
  data_dir = params.data_dir
  if hasattr(params, "rand_state"):
    rand_state = params.rand_state
  else:
    assert hasattr(params, "rand_seed"), ("Params must specify a random state or seed")
    rand_state = np.random.RandomState(params.rand_seed)
  num_batches = int(params.num_batches) if hasattr(params, "num_batches") else None
  ti_data = tinyImages(data_dir, num_batches, rand_state)
  pf.plot_data_tiled(ti_data.images[1000:1100,...], cmap=None, save_filename="./tmp.png")
  image_dataset = Dataset(ti_data.images, lbls=None, ignore_lbls=None, rand_state=rand_state)
  return {"train":image_dataset}
