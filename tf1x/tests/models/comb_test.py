import copy
import os
import sys

root_path = os.path.dirname(os.path.dirname(os.getcwd()))
if root_path not in sys.path: sys.path.append(root_path)

import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.params.param_picker as pp
import DeepSparseCoding.tf1x.models.model_picker as mp
import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.utils.data_processing as dp

"""
Test for running models
NOTE: Should be executed from the repository's root directory
loads every model
loads every dataset

for each model:
  modify model params to be simple
  for each dataset:
    if should_fail:
      print warning
    else:
      do stuff
    #try:
    #  do preprocessing
    #  get batch
    #  do one weight update step
    #catch exception:
    #  if should_fail:
    #    print "warning: model with dataset failed" + str(exception)
    #    raise NotImplementedError
    #  else:
    #    assert False, "model with dataset failed"
## TODO:
  * rica model has different interface for applying gradients when the L-BFGS minimizer is used
    this is inconsistent and should be changed so that it acts like all models
"""

def testBasic(self):
  should_fail_list = [
    ("mlp", "vanHateren"),
    ("mlp", "field"),
    ("mlp_lca", "CIFAR10"),
    ("mlp_lca", "vanHateren"),
    ("mlp_lca", "field"),
    ("mlp_lca_subspace", "CIFAR10"),
    ("mlp_lca_subspace", "vanHateren"),
    ("mlp_lca_subspace", "field"),
    ("mlp_ae", "CIFAR10"),
    ("mlp_ae", "vanHateren"),
    ("mlp_ae", "field"),
    ("mlp_sae", "CIFAR10"),
    ("mlp_sae", "vanHateren"),
    ("mlp_sae", "field"),
    ("mlp_vae", "CIFAR10"),
    ("mlp_vae", "vanHateren"),
    ("mlp_vae", "field"),
    ("mlp_lista", "CIFAR10"),
    ("mlp_lista", "vanHateren"),
    ("mlp_lista", "field"),
    ("ica_pca", "field"),
    ("ica_pca", "MNIST"),
    ("ica_pca", "CIFAR10"),
    ("rica", "CIFAR10"),
    ("lca", "CIFAR10"),
    ("lca_pca", "CIFAR10"),
    ("lca_pca_fb", "CIFAR10"),
    ("lca_subspace", "CIFAR10"),
    ("sae", "CIFAR10"),
    ("lista", "vanHateren"),
    ("lista", "field"),
    ("lista", "CIFAR10"),
    ("dae", "field"),
    ("dae", "vanHateren"),
    ("dae_mem", "field"),
    ("dae_mem", "CIFAR10"),
    ("dae_mem", "vanHateren"),
    ("vae", "field"),
    ("vae", "CIFAR10"),
    ("lambda", "vanHateren"),
    ("lambda", "field")]

  model_should_fail = False
  if (self.model_type, self.data_type) in should_fail_list:
    model_should_fail = True
  if model_should_fail:
    print("Model "+self.model_type+" failed with dataset "+self.data_type+", as expected.")
  else:
    schedule_index = 0 # TODO: test support for multiple schedules
    params = pp.get_params(self.model_type) # Import params
    params.set_test_params(self.data_type)
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
      params.model_name = "test_all_"+params.model_name
      params.out_dir += "tests/"
      params.data_type = self.data_type
      model = mp.get_model(self.model_type) # Import model
      model.data_type = self.data_type
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
        ## FIXME ##
        if self.model_type == "rica" and hasattr(model, "minimizer"):
          model.minimizer.minimize(session=sess, feed_dict=feed_dict)

model_list = mp.get_model_list()
data_list = ds.get_dataset_list()
for model_type in model_list:
  for data_type in data_list:
    #Define class name with model_type
    class_name = "AllTest_"+str(model_type)+"_"+str(data_type)
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
