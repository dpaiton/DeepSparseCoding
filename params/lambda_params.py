import os
import tensorflow as tf
from modules.activations import activation_picker
from params.base_params import BaseParams

class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "lambda"
    self.model_name = "lambda"
    self.version = "0.0"
    self.vectorize_data = True
    self.standardize_data = False
    self.batch_size = 100
    self.max_cp_to_keep = 1
    self.val_on_cp = True
    self.eval_batch_size = 1000
    self.cp_load = False
    self.log_to_file = True
    self.save_plots = True
    self.val_on_cp = False
    self.cp_int = 10000
    self.log_int = 100
    self.gen_plot_int = 1e4
    self.activation_function = tf.identity
    self.schedule = [
      {"num_batches": 1,
      "weights": None,
      "weight_lr": 0.01,
      "decay_steps": int(1e4*0.5),
      "decay_rate": 0.8,
      "staircase": True}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "mnist":
      self.model_name += "_mnist"
      self.vectorize_data = True
      self.rescale_data = False
      self.center_data = False
      self.whiten_data = False
      self.standardize_data = True
      self.num_val = 10000

    elif data_type.lower() == "cifar10":
      self.model_name += "_cifar10"
      self.vectorize_data = True
      self.rescale_data = False
      self.standardize_data = False
      self.center_data = False
      self.whiten_data = True
      self.extract_patches = False
      self.log_int = 100
      self.cp_int = 500
      self.gen_plot_int = 1e3

    elif data_type.lower() == "synthetic":
      self.model_name += "_synthetic"
      self.epoch_size = 1000
      self.dist_type = "gaussian"
      self.num_edge_pixels = 16
      self.num_classes = 2

    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.set_data_params(data_type)
    self.vectorize_data = True
    self.epoch_size = 50
    self.batch_size = 50
    self.num_edge_pixels = 8
