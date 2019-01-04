import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp
import sys

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
    ("mlp", "CIFAR10"),
    ("mlp", "vanHateren"),
    ("mlp", "field"),
    ("mlp", "tinyImages"),
    ("mlp", "synthetic"),
    ("ica", "vanHateren"),
    ("ica", "field"),
    ("ica", "MNIST"),
    ("ica", "CIFAR10"),
    ("ica", "tinyImages"),
    ("ica", "synthetic"),
    ("ica_pca", "vanHateren"),
    ("ica_pca", "field"),
    ("ica_pca", "MNIST"),
    ("ica_pca", "CIFAR10"),
    ("ica_pca", "tinyImages"),
    ("ica_pca", "synthetic"),
    ("rica", "vanHateren"),
    ("rica", "field"),
    ("rica", "MNIST"),
    ("rica", "CIFAR10"),
    ("rica", "tinyImages"),
    ("rica", "synthetic"),
    ("lca", "CIFAR10"),
    ("lca", "tinyImages"),
    ("lca_pca", "vanHateren"),
    ("lca_pca", "field"),
    ("lca_pca", "MNIST"),
    ("lca_pca", "CIFAR10"),
    ("lca_pca", "tinyImages"),
    ("lca_pca", "synthetic"),
    ("lca_pca_fb", "vanHateren"),
    ("lca_pca_fb", "field"),
    ("lca_pca_fb", "MNIST"),
    ("lca_pca_fb", "CIFAR10"),
    ("lca_pca_fb", "tinyImages"),
    ("lca_pca_fb", "synthetic"),
    ("lca_conv", "tinyImages"),
    ("lca_subspace", "field"),
    ("lca_subspace", "CIFAR10"),
    ("lca_subspace", "tinyImages"),
    ("lca_subspace", "synthetic"),
    ("sigmoid_autoencoder", "CIFAR10"),
    ("sigmoid_autoencoder", "tinyImages"),
    ("gdn_autoencoder", "vanHateren"),
    ("gdn_autoencoder", "field"),
    ("gdn_autoencoder", "MNIST"),
    ("gdn_autoencoder", "CIFAR10"),
    ("gdn_autoencoder", "tinyImages"),
    ("gdn_autoencoder", "synthetic"),
    ("gdn_conv_autoencoder", "vanHateren"),
    ("gdn_conv_autoencoder", "field"),
    ("gdn_conv_autoencoder", "MNIST"),
    ("gdn_conv_autoencoder", "CIFAR10"),
    ("gdn_conv_autoencoder", "tinyImages"),
    ("gdn_conv_autoencoder", "synthetic"),
    ("gdn_conv_decoder", "vanHateren"),
    ("gdn_conv_decoder", "field"),
    ("gdn_conv_decoder", "MNIST"),
    ("gdn_conv_decoder", "CIFAR10"),
    ("gdn_conv_decoder", "tinyImages"),
    ("gdn_conv_decoder", "synthetic"),
    ("relu_autoencoder", "vanHateren"),
    ("relu_autoencoder", "field"),
    ("relu_autoencoder", "MNIST"),
    ("relu_autoencoder", "CIFAR10"),
    ("relu_autoencoder", "tinyImages"),
    ("relu_autoencoder", "synthetic"),
    ("fista", "vanHateren"),
    ("fista", "field"),
    ("fista", "MNIST"),
    ("fista", "CIFAR10"),
    ("fista", "tinyImages"),
    ("fista", "synthetic"),
    ("vae", "vanHateren"),
    ("vae", "field"),
    ("vae", "CIFAR10"),
    ("vae", "tinyImages"),
    ("vae", "synthetic")]

  schedule_index = 0 # Not testing support for multiple schedules

  params = pp.get_params(self.model_type) # Import params
  model = mp.get_model(self.model_type) # Import model
  params.data_type = self.data_type
  model.data_type = self.data_type

  model.should_fail = False
  if (self.model_type, self.data_type) in should_fail_list:
    model.should_fail = True

  if model.should_fail:
    print("Model "+self.model_type+" failed with dataset "+self.data_type+", as expected.")
  else:
    params.set_test_params(self.data_type)
    params.model_name = "test_all_"+params.model_name
    dataset = ds.get_data(params) # Import data
    dataset = model.preprocess_dataset(dataset, params)
    dataset = model.reshape_dataset(dataset, params)
    params.data_shape = list(dataset["train"].shape[1:])
    model.setup(params)
    model.write_saver_defs()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
      sess.run(model.init_op,
        feed_dict={model.x:np.zeros([params.batch_size]+params.data_shape, dtype=np.float32)})
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
