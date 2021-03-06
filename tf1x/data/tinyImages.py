import numpy as np

from DeepSparseCoding.tf1x.data.dataset import Dataset
import DeepSparseCoding.tf1x.utils.plot_functions as pf

class tinyImages(object):
  def __init__(self, image_dir, num_epochs=1, epoch_size=1000, rand_state=np.random.RandomState()):
    self.total_num_images = num_epochs * epoch_size
    self.image_shape = [32, 32, 3]
    self.num_pixels = int(np.prod(self.image_shape))
    start_index = 0
    self.images = self.extract_random_images(image_dir, num_epochs, epoch_size, start_index,
      rand_state)

  def extract_random_images(self, image_dir, num_epochs, epoch_size, start_index, rand_state):
    self.indices = rand_state.permutation(self.total_num_images)
    file_location = image_dir+"tiny_images.bin"
    with open(file_location, "rb") as data_file:
      images = []
      for img_index in range(start_index, int((num_epochs*epoch_size)-start_index), epoch_size):
        offset = img_index * self.num_pixels
        data_file.seek(offset)
        data = np.frombuffer(data_file.read(self.num_pixels*epoch_size), dtype='uint8')
        data = np.stack(np.split(data, epoch_size))
        images.append(data.reshape([epoch_size]+self.image_shape, order="F"))
    return np.concatenate(images, axis=0)

def load_tinyImages(params):
  data_dir = params.data_dir
  if hasattr(params, "rand_state"):
    rand_state = params.rand_state
  else:
    assert hasattr(params, "rand_seed"), ("Params must specify a random state or seed")
    rand_state = np.random.RandomState(params.rand_seed)
  num_epochs = int(params.num_epochs) # required parameter
  epoch_size = int(params.epoch_size) # required parameter
  ti_data = tinyImages(data_dir, num_epochs, epoch_size, rand_state)
  image_dataset = Dataset(ti_data.images, lbls=None, ignore_lbls=None, rand_state=rand_state)
  return {"train":image_dataset}
