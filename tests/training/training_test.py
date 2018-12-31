import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp

"""
Test for training models
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
  * if model/dataset combo doesn't work, then test model loading alone
  * test all models for each dataset, load datasets ahead of time? weird because of parameter ordering
  * rica model has different interface for applying gradients when the L-BFGS minimizer is used
    this is inconsistent and should be changed so that it acts like all models
"""
class TrainingTest(tf.test.TestCase):
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
      ("lca", "synthetic"),
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
      ("conv_lca", "vanHateren"),
      ("conv_lca", "field"),
      ("conv_lca", "MNIST"),
      ("conv_lca", "CIFAR10"),
      ("conv_lca", "tinyImages"),
      ("conv_lca", "synthetic"),
      ("sigmoid_autoencoder", "vanHateren"),
      ("sigmoid_autoencoder", "field"),
      ("sigmoid_autoencoder", "MNIST"),
      ("sigmoid_autoencoder", "CIFAR10"),
      ("sigmoid_autoencoder", "tinyImages"),
      ("sigmoid_autoencoder", "synthetic"),
      ("gdn_autoencoder", "vanHateren"),
      ("gdn_autoencoder", "field"),
      ("gdn_autoencoder", "MNIST"),
      ("gdn_autoencoder", "CIFAR10"),
      ("gdn_autoencoder", "tinyImages"),
      ("gdn_autoencoder", "synthetic"),
      ("conv_gdn_autoencoder", "vanHateren"),
      ("conv_gdn_autoencoder", "field"),
      ("conv_gdn_autoencoder", "MNIST"),
      ("conv_gdn_autoencoder", "CIFAR10"),
      ("conv_gdn_autoencoder", "tinyImages"),
      ("conv_gdn_autoencoder", "synthetic"),
      ("conv_gdn_decoder", "vanHateren"),
      ("conv_gdn_decoder", "field"),
      ("conv_gdn_decoder", "MNIST"),
      ("conv_gdn_decoder", "CIFAR10"),
      ("conv_gdn_decoder", "tinyImages"),
      ("conv_gdn_decoder", "synthetic"),
      ("relu_autoencoder", "vanHateren"),
      ("relu_autoencoder", "field"),
      ("relu_autoencoder", "MNIST"),
      ("relu_autoencoder", "CIFAR10"),
      ("relu_autoencoder", "tinyImages"),
      ("relu_autoencoder", "synthetic"),
      ("vae", "vanHateren"),
      ("vae", "field"),
      ("vae", "CIFAR10"),
      ("vae", "tinyImages"),
      ("vae", "synthetic")]

    model_list = mp.get_model_list()
    data_list = ds.get_dataset_list()
    schedule_index = 0 # Not testing support for multiple schedules

    for model_type in model_list:
      for data_type in data_list:
        params = pp.get_params(model_type) # Import params
        model = mp.get_model(model_type) # Import model
        params.data_type = data_type
        model.data_type = data_type

        model.should_fail = False
        if (model_type, data_type) in should_fail_list:
          model.should_fail = True

        if model.should_fail:
          print("Model "+model_type+" failed with dataset "+data_type+", as expected.")
        else:
          try:
            params.set_data_params(data_type)
            params.model_name = "training_test_"+params.model_name
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
              if model_type == "rica" and hasattr(model, "minimizer"):
                model.minimizer.minimize(session=sess, feed_dict=feed_dict)
              ####
              print("Model "+model_type+" passed on dataset "+data_type+".")
          except Exception as e:
            #import IPython; IPython.embed(); raise SystemExit
            print("Model "+model_type+" failed on dataset "+data_type)
            raise e

if __name__ == "__main__":
  tf.test.main()
