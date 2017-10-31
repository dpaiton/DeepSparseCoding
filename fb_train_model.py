import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds

## Specify training run params
model_type = "lca_pca_fb"
data_type = "vanhateren"
cov_suffix = "300k_imgs"

## Import params
params, schedule = pp.get_params(model_type)
if "rand_seed" in params.keys():
  params["rand_state"] = np.random.RandomState(params["rand_seed"])
params["data_type"] = data_type
params["cov_suffix"] = cov_suffix

## Import data
data = ds.get_data(params)
if params["conv"]: # conv param is set in the param picker
  params["input_shape"] = [data["train"].num_rows, data["train"].num_cols,
    data["train"].num_channels]
  for key in data.keys():
      data[key].devectorize_data()
else:
  params["input_shape"] = [
    data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]
params["num_pixels"] = data["train"].num_pixels

## Import model
model = mp.get_model(model_type, params, schedule)

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  # Need to provide shape if batch_size is used in graph
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros([params["batch_size"]]+params["input_shape"],
    dtype=np.float32)})

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  # Load model weights from pretrained session
  model.load_weights(sess, tf.train.latest_checkpoint(model.cp_load_dir))

  # Fine tune weights using feedback
  model.log_info("Beginning schedule "+str(model.sched_idx))
  for b_step in range(model.get_sched("num_batches")):
    data_batch = data["train"].next_batch(model.batch_size)
    input_data = data_batch[0]
    input_labels = data_batch[1]
    feed_dict = model.get_feed_dict(input_data, input_labels)

    ## Normalize weights
    if hasattr(model, "norm_weights"):
      if params["norm_weights"]:
        sess.run([model.norm_weights], feed_dict)

    # Reset activity from previous batch
    if hasattr(model, "reset_activity"):
      sess.run([model.reset_activity], feed_dict)

    ## Update weights
    for w_idx in range(len(model.get_sched("weights"))):
      sess.run(model.apply_grads[model.sched_idx][w_idx], feed_dict)

    ## Generate logs
    current_step = sess.run(model.global_step)
    if (current_step % model.log_int == 0
      and model.log_int > 0):
      model.print_update(input_data=input_data, input_labels=input_labels,
        batch_step=b_step+1)

    ## Plot weights & gradients
    if (current_step % model.gen_plot_int == 0
      and model.gen_plot_int > 0):
      model.generate_plots(input_data=input_data, input_labels=input_labels)

    ## Checkpoint
    if (current_step % model.cp_int == 0
      and model.cp_int > 0):
      save_dir = model.write_checkpoint(sess)

  save_dir = model.write_checkpoint(sess)
  print("Training Complete\n")
