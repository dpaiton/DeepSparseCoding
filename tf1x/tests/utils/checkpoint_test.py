import numpy as np
import tensorflow as tf
import utils.data_processing as dp
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds
import pdb

"""
Test for checkpointing
"""
class CheckpointTest(tf.test.TestCase):
  def buildModel(self, model_name_prefix, load_style="none"):
    model_type = "mlp"
    data_type = "synthetic"

    params = pp.get_params(model_type) # Import params
    params.model_name = "mlp"
    params.cp_int = 1
    params.batch_norm = [False, False]
    params.schedule[0]["weight_lr"] = 100

    if(load_style=="all"):
      #TODO
      params.cp_load = True
      params.cp_load_name = "test_checkpoints_mlp_synthetic"
      params.cp_load_step = 1
      params.cp_load_ver = "0.0"
      params.cp_load_var = None
    elif(load_style=="partial"):
      #TODO
      params.cp_load = True
      params.cp_load_name = "test_checkpoints_mlp_synthetic"
      params.cp_load_step = 1
      params.cp_load_ver = "0.0"
      params.cp_load_var = ["mlp/layer0/conv_w_0:0", "mlp/layer0/conv_b_0:0"]
    elif(load_style=="none"):
      pass
    else:
      assert(False)

    model = mp.get_model(model_type) # Import model
    params.data_type = data_type
    model.data_type = data_type
    params.set_test_params(data_type)

    params.model_name = model_name_prefix+params.model_name
    params.out_dir = params.out_dir+"/tests/"
    dataset = ds.get_data(params) # Import data
    dataset = model.preprocess_dataset(dataset, params)
    dataset = model.reshape_dataset(dataset, params)
    params.data_shape = list(dataset["train"].shape[1:])
    model.setup(params)
    model.write_saver_defs()
    return (params, dataset, model)

  def testBasic(self):
    params, dataset, model = self.buildModel("test_checkpoints_", load_style="none")
    schedule_index = 0 # Not testing support for multiple schedules

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=model.graph) as sess:
      sess.run(model.init_op,
        feed_dict={model.input_placeholder:np.zeros([params.batch_size]+params.data_shape, dtype=np.float32)})
      sess.graph.finalize() # Graph is read-only after this statement
      model.write_graph(sess.graph_def)
      model.sched_idx = 0
      data, labels, ignore_labels  = dataset["train"].next_batch(model.params.batch_size)
      feed_dict = model.get_feed_dict(data, labels)
      for w_idx in range(len(model.get_schedule("weights"))):
        sess.run(model.apply_grads[schedule_index][w_idx], feed_dict)

      #Grab weights for check
      pre_vars = {}
      for name, node in model.trainable_variables.items():
        pre_vars[name] = sess.run(node)

      #Write a checkpoint
      model.write_checkpoint(sess)

    #Test loading all variables
    params, dataset, model = self.buildModel("test_checkpoints_", load_style="all")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=model.graph) as sess:
      sess.run(model.init_op,
        feed_dict={model.input_placeholder:np.zeros([params.batch_size]+params.data_shape, dtype=np.float32)})
      sess.graph.finalize() # Graph is read-only after this statement

      cp_load_file = (params.cp_load_dir+params.cp_load_name+"_v"+params.cp_load_ver
        +"-"+str(params.cp_load_step))
      model.load_checkpoint(sess, cp_load_file)

      post_vars = {}
      for name, node in model.trainable_variables.items():
        post_vars[name] = sess.run(node)

    cmp_eps = 1e-10
    for name in pre_vars.keys():
      pre_val = pre_vars[name]
      post_val = post_vars[name]
      self.assertAllClose(pre_val, post_val, rtol=cmp_eps, atol=cmp_eps)

    #Test loading specific variables
    #Only loading in first layer weights and biases
    params, dataset, model = self.buildModel("test_checkpoints_", load_style="partial")
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=model.graph) as sess:
      sess.run(model.init_op,
        feed_dict={model.input_placeholder:np.zeros([params.batch_size]+params.data_shape, dtype=np.float32)})
      sess.graph.finalize() # Graph is read-only after this statement

      cp_load_file = (params.cp_load_dir+params.cp_load_name+"_v"+params.cp_load_ver
        +"-"+str(params.cp_load_step))
      model.load_checkpoint(sess,cp_load_file)

      post_vars = {}
      for name, node in model.trainable_variables.items():
        post_vars[name] = sess.run(node)

    for name in pre_vars.keys():
      pre_val = pre_vars[name]
      post_val = post_vars[name]
      if(name in params.cp_load_var):
        self.assertAllClose(pre_val, post_val, rtol=cmp_eps, atol=cmp_eps)
      else:
        #Ignore biases and batch norm since they don't change much in one weight update
        if("/b_" in name or "/BatchNorm" in name or "_b_" in name):
          pass
        else:
          self.assertNotAllClose(pre_val, post_val, rtol=cmp_eps, atol=cmp_eps)

    print("Checkpoint Test Passed")

if __name__ == "__main__":
  tf.test.main()
