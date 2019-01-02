import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import utils.data_processing as dp

"""
Test for running models
NOTE: Should be executed from the repository's root directory
loads every model and runs on synthetic data

## TODO:
  * rica model has different interface for applying gradients when the L-BFGS minimizer is used
    this is inconsistent and should be changed so that it acts like all models
"""
class RunTest(tf.test.TestCase):
  def testBasic(self):
    model_list = ["mlp", "vae", "lca", "lca_conv",
      "lca_pca", "lca_subspace"]#mp.get_model_list()
    data_type = "synthetic"
    schedule_index = 0 # Not testing support for multiple schedules

    for model_type in model_list:
      params = pp.get_params(model_type) # Import params
      model = mp.get_model(model_type) # Import model
      params.data_type = data_type
      model.data_type = data_type

      try:
        params.set_data_params(data_type)
        params.model_name = "test_run_"+params.model_name
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
        raise Exception("Model "+model_type+" on RunTest failed:\n") from e

if __name__ == "__main__":
  tf.test.main()
