import os 
import numpy as np 

class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "ica_subspace"
    self.model_name = "ica_subspace"
    self.version = "0.0"

    self.num_groups = 5
    self.group_sizes = None


