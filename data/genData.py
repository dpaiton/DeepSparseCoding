import numpy as np
from data.dataset import Dataset

class DistData(object): 
  def __init__(self,
    num_pixels,
    num_examples,
    rand_state = np.random.RandomState()):
 
    self.num_edge_pixels = int(np.sqrt(num_pixels))
    self.num_examples = num_examples

def load_dist(kwargs):
  data = DistData(kwargs["num_pixels"], kwargs["num_examples"])
  dist_type = kwargs["dist_type"]
  rand_seed = kwargs["rand_seed"]
  if dist_type == "gaussian":
    d = np.array([[[np.random.normal(kwargs["a"], kwargs["b"]) 
      for k in range(data.num_edge_pixels)] for j in range(data.num_edge_pixels)] 
      for i in range(data.num_examples)], dtype=np.float32)
  if dist_type == "laplacian":
    d = np.array([[[np.random.laplace(kwargs["a"], kwargs["b"]) 
      for k in range(data.num_edge_pixels)] for j in range(data.num_edge_pixels)] 
      for i in range(data.num_examples)], dtype=np.float32)
  data = Dataset(d, None, None)
  return {"train": data}