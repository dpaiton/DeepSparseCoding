from params.ica_params import params as ica_params

import os 
import numpy as np 


class params(ica_params):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "ica_subspace"
    self.model_name = "ica_subspace"
    self.version = "0.0"

    self.num_groups = 5
    self.group_sizes = None

    self.center_data = True
    self.whiten_data = True
    self.whiten_method = "PCA"


