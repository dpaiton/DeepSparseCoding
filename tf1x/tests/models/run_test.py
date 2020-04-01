import copy
import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp
import sys

"""
Test for running models
loads every model and runs on synthetic data
"""
def testBasic(self):
  schedule_index = 0 # Not testing support for multiple schedules
  params = pp.get_params(self.model_type) # Import params
  params.set_test_params(self.data_type)
  original_schedule = copy.deepcopy(params.schedule)
  if not hasattr(params, "test_param_variants"):
    params.test_param_variants = list(dict())
  for variant_number, variant in enumerate(params.test_param_variants):
    print("Test variant number "+str(variant_number))
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
    params.model_name = "test_run_"+params.model_name
    params.out_dir += "tests/"
    dataset = ds.get_data(params) # Import data
    dataset = model.preprocess_dataset(dataset, params)
    dataset = model.reshape_dataset(dataset, params)
    params.data_shape = list(dataset["train"].shape[1:])
    model.setup(params)
    model.write_saver_defs()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=model.graph) as sess:
      sess.run(model.init_op,
        feed_dict={model.input_placeholder:np.zeros([params.batch_size]+params.data_shape,
        dtype=np.float32)})
      sess.graph.finalize() # Graph is read-only after this statement
      model.write_graph(sess.graph_def)
      model.sched_idx = 0
      data, labels, ignore_labels  = dataset["train"].next_batch(model.params.batch_size)
      feed_dict = model.get_feed_dict(data, labels)
      for w_idx in range(len(model.get_schedule("weights"))):
        sess.run(model.apply_grads[schedule_index][w_idx], feed_dict)
      if model.params.optimizer == "lbfgsb":
        model.minimizer.minimize(session=sess, feed_dict=feed_dict)


#Make class with specific model_type name in class name
model_list = mp.get_model_list()
data_type = "synthetic"
for model_type in model_list:
  #Define class name with model_type
  class_name = "RunTest_"+str(model_type)
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
