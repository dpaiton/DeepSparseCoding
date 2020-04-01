import os

import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.data.data_selector as ds

class params(object):
  def __init__(self):
    self.data_dir = os.path.expanduser("~")+"/Work/Datasets/"
    self.num_epochs = 2 # Tiny Images
    self.epoch_size = 100 # Tiny Images
    self.num_val = 1000 # CIFAR / MNIST
    self.dist_type = "gaussian" # Synthetic
    self.num_edge_pixels = 16 # Synthetic
    self.rand_seed = 123456789
    self.rand_state = np.random.RandomState(self.rand_seed)

class DataSelectorTest(tf.test.TestCase):
  def testBasic(self):
    ## Test that all datatypes can load
    ## Test that the random draw is the same for each instance
    for data_type in ds.get_dataset_list():
      params1 = params()
      params1.data_type = data_type

      data_draw1 = ds.get_data(params1)

      params2 = params()
      params2.data_type = data_type
      data_draw2 = ds.get_data(params2)

      self.assertAllEqual(data_draw1["train"].images, data_draw2["train"].images)
      print(data_type+" test passed.")

if __name__ == "__main__":
  tf.test.main()
