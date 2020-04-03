import copy
import sys

import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.params.param_picker as pp
import DeepSparseCoding.tf1x.models.model_picker as mp
import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.utils.data_processing as dp

"""
Test for building models
NOTE: Should be executed from the repository's root directory
loads every model
"""
#Base method each dynamically generated class requires
def testBasic(self):
  schedule_index = 0 # Not testing support for multiple schedules
  params = pp.get_params(self.model_type) # Import params
  params.set_test_params(self.data_type)
  #Don't do whitening in params for build
  params.whiten_data = False
  params.lpf_data = False
  original_schedule = copy.deepcopy(params.schedule)
  if not hasattr(params, "test_param_variants"):
    params.test_param_variants = list(dict())
  for variant_number, variant in enumerate(params.test_param_variants):
    print("Test variant "+str(variant_number))
    for key in variant.keys():
      if key == "posterior_prior": # VAE
        prior_params = {
          "posterior_prior":variant[key],
          "gauss_mean":0.0,
          "gauss_std":1.0,
          "cauchy_location":0.0,
          "cauchy_scale":1.0,
          "laplace_scale":1.0
        }
        setattr(params, "prior_params", prior_params)
      else:
        setattr(params, key, variant[key])
    params.schedule = copy.deepcopy(original_schedule)
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
model_list = mp.get_model_list()
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
