import numpy as np
from data.dataset import Dataset
from scipy.stats import norm

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

"""
Generate sparse hierarchical dataset
"""
def gen_hier_data(kwargs):
  I = kwargs["num_pixels"]
  N = kwargs["num_examples"]

  # Number of second-level basis functions
  J = 10 

  # First-level identity basis
  A = np.identity(I)

  # Second-level Gaussian basis
  sig = int(np.sqrt(I)/2)
  B = np.asarray([norm.pdf(np.arange(0, I), np.random.randint(0, I), sig) 
    for i in range(J)], dtype = np.float32).T
  B /= (np.max(B, axis = 0)[None,:])

  # Laplacian prior over B
  v = np.random.laplace(0, 1, (J, N))
  lam = np.exp(B @ v)
  u = np.sqrt(lam) * np.random.normal(0, 1, (I, N))

  # Generate data
  x = np.reshape(A @ u, (N, int(np.sqrt(I)), int(np.sqrt(I))))
  data = Dataset(x, None, None)
  return {"train": data}
