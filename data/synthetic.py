import numpy as np
from data.dataset import Dataset
import utils.data_processing as dp

class synthetic(object):
  def __init__(self, dist_type, epoch_size, num_edge_pixels, rand_state=np.random.RandomState()):
    self.dist_type = dist_type
    self.epoch_size = epoch_size
    self.num_edge_pixels = num_edge_pixels
    self.rand_state = rand_state
    self.images = self.generate_data()
    self.labels = self.generate_labels()

  def generate_data(self):
    if self.dist_type == "gaussian":
      data = self.rand_state.normal(loc=0.0, scale=1.0,
        size=(self.epoch_size, self.num_edge_pixels, self.num_edge_pixels, 1))
    elif self.dist_type == "laplacian":
      data = self.rand_state.laplace(loc=0.0, scale=1.0,
        size=(self.epoch_size, self.num_edge_pixels, self.num_edge_pixels, 1))
    else:
      data = np.zeros((self.epoch_size, self.num_edge_pixels, self.num_edge_pixels, 1))
    return data

  def generate_labels(self):
    #Generate classes from 2 avaliable classes
    num_classes = 2
    labels = np.random.randint(num_classes, size=self.epoch_size)
    labels = dp.dense_to_one_hot(labels, num_classes)
    return labels

def load_synthetic(params):
  if hasattr(params, "rand_state"):
    rand_state = params.rand_state
  else:
    assert hasattr(params, "rand_seed"), ("Params must specify a random state or seed")
    rand_state = np.random.RandomState(params.rand_seed)
  synth_data = synthetic(
    dist_type=params.dist_type,
    num_edge_pixels=params.num_edge_pixels,
    epoch_size=params.epoch_size,
    rand_state=params.rand_state)
  data = Dataset(synth_data.images, lbls=synth_data.labels, ignore_lbls=None, rand_state=rand_state)
  return {"train": data}
