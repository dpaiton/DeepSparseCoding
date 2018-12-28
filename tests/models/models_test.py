import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp

"""
Test for building models
NOTE: Should be executed from the repository's root directory
loads every model
"""
class ModelsTest(tf.test.TestCase):
  def testBasic(self):
    data_type = "mnist"
    model_list = ["mlp", "vae"]#mp.get_model_list()
    schedule_index = 0 # Not testing support for multiple schedules

    for model_type in model_list:
      params = pp.get_params(model_type) # Import params
      model = mp.get_model(model_type) # Import model
      params.data_type = data_type
      model.data_type = data_type
      params.model_name = "models_test_" + params.model_name
      try:
        #params.set_test_params(data_type)
        dataset = ds.get_data(params) # Import data
        dataset = model.preprocess_dataset(dataset, params)
        dataset = model.reshape_dataset(dataset, params)
        params.data_shape = list(dataset["train"].shape[1:])
        #params.set_test_params(data_type)
        model.setup(params)
        print("Model "+model_type+" passed")
      except Exception as e:
        #import IPython; IPython.embed(); raise SystemExit
        print("Model "+model_type+" failed with following error:\n")
        raise e

if __name__ == "__main__":
  tf.test.main()
