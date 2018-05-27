import numpy as np
from data.dataset import Dataset

class synthetic(object):
  def __init__(self, dist_type, epoch_size, num_edge_pixels, rand_state=np.random.RandomState()):
    self.dist_type = dist_type
    self.epoch_size = epoch_size
    self.num_edge_pixels = num_edge_pixels
    self.rand_state = rand_state
    self.images = self.generate_data()

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

def load_synthetic(kwargs):
  rand_state = kwargs["rand_state"] if "rand_state" in kwargs.keys() else np.random.RandomState()
  synth_data = synthetic(
    dist_type=kwargs["dist_type"],
    num_edge_pixels=kwargs["patch_edge_size"],
    epoch_size=kwargs["epoch_size"],
    rand_state=kwargs["rand_state"])
  data = Dataset(synth_data.images, lbls=None, ignore_lbls=None, rand_state=rand_state)
  return {"train": data}
