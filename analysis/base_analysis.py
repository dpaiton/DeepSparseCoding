import os
import numpy as np
import matplotlib.pyplot as plt
import utils.plot_functions as pf
import utils.log_parser
import models.model_picker as mp
from data.MNIST import load_MNIST
import tensorflow as tf

class Analyzer(object):
  def __init__(self, params):
    self.load_params(params)
    self.make_dirs()

  def load_params(self, params):
    # Model details
    self.version = params["version"]
    self.model_name = params["model_name"]
    self.batch_index = params["batch_index"]
    # Analysis details
    self.eval_train = params["eval_train"]
    self.eval_test = params["eval_test"]
    self.eval_val = params["eval_val"]
    # Output details
    self.file_ext = params["file_ext"]
    self.device = params["device"]
    self.out_dir = (params["out_dir"]+"/"+self.model_name+"/"
      +self.version+"/")

  """Make output directories"""
  def make_dirs(self):
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)
