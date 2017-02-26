import numpy as np
from data.dataset import Dataset

class synthetic(object):
  def __init__(self, dist_type, num_examples, num_edge_pixels, rand_state):
    self.dist_type = dist_type
    self.num_examples = num_examples
    self.num_edge_pixels = num_edge_pixels
    self.rand_state = rand_state
    self.images = self.generate_data()

  def generate_data(self):
    if self.dist_type == "gaussian":
      data = self.rand_state.normal(loc=0.0, scale=1.0,
        size=(self.num_examples, self.num_edge_pixels, self.num_edge_pixels))
    elif self.dist_type == "laplacian":
      data = self.rand_state.laplace(loc=0.0, scale=1.0,
        size=(self.num_examples, self.num_edge_pixels, self.num_edge_pixels))
    else:
      data = np.zeros((self.num_examples, self.num_edge_pixels,
        self.num_edge_pixels))
    return data

def load_synthetic(kwargs):
  if "rand_state" not in kwargs.keys():
    if "rand_seed" not in kwargs.keys():
      kwargs["rand_state"] = np.random.RandomState()
    else:
      kwargs["rand_state"] = np.random.RandomState(kwargs["rand_seed"])
  synth_data = synthetic(
    dist_type=kwargs["dist_type"],
    num_edge_pixels=np.int(np.sqrt(kwargs["num_pixels"])),
    num_examples=kwargs["num_examples"],
    rand_state=kwargs["rand_state"])
  data = Dataset(synth_data.images, None, None)
  return {"train": data}
