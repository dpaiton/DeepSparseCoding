import os
import numpy as np
import data.data_selector as ds

class params(object):
  def __init__(self):
    self.data_type = "mnist"
    self.data_dir = os.path.expanduser("~")+"/Work/Datasets/"
    self.num_val = 10000
    self.rand_seed = 123456789
    self.rand_state = np.random.RandomState(self.rand_seed)

def test_mnist():
  mnist_params1 = params()
  mnist_data_draw1 = ds.get_data(mnist_params1)

  mnist_params2 = params()
  mnist_data_draw2 = ds.get_data(mnist_params2)

  assert np.all(mnist_data_draw1["train"].images == mnist_data_draw2["train"].images)


if __name__ == "__main__":
  test_mnist()
