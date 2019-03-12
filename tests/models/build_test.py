import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp
import sys

"""
Test for building models
NOTE: Should be executed from the repository's root directory
loads every model
TODO: allow test_params to hvae multiple entries & test each entry
"""

#Base method each dynamically generated class requires
def testBasic(self):
  schedule_index = 0 # Not testing support for multiple schedules
  params = pp.get_params(self.model_type) # Import params
  params.set_test_params(self.data_type)
  #Don't do whitening in params for build
  params.whiten_data = False
  params.lpf_data = False

  model = mp.get_model(self.model_type) # Import model
  params.data_type = self.data_type
  model.data_type = self.data_type
  params.model_name = "test_build_" + params.model_name
  params.out_dir += "tests/"
  dataset = ds.get_data(params) # Import data
  dataset = model.preprocess_dataset(dataset, params)
  dataset = model.reshape_dataset(dataset, params)
  params.data_shape = list(dataset["train"].shape[1:])

  model.setup(params)

#Make class with specific model_type name in class name
#model_list = mp.get_model_list()
model_list = ["mlp", "mlp_lca", "mlp_vae", "mlp_sae", "mlp_lista", "ae", "vae", "lca", "lca_conv", "lca_pca", "lca_pca_fb", "lca_subspace", "fflista", "lista", "ica", "sae", "dae", "rica"]
data_type = "synthetic"
for model_type in model_list:
  #Define class name with model_type
  class_name = "BuildTest_"+str(model_type)
  #Define class with class name, inherited from tf.test.TestCase
  #and define attributes for the class
  class_def = type(class_name, (tf.test.TestCase,),
    {"testBasic": testBasic,
    "model_type": model_type,
    "data_type": data_type})
  #Add this to module names so import * imports these class names
  setattr(sys.modules[__name__], class_name, class_def)

if __name__ == "__main__":
  tf.test.main()
