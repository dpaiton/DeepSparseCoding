import matplotlib
matplotlib.use("Agg")

import time as ti
t0 = ti.time()
import numpy as np
import tensorflow as tf
import json as js
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds

## Specify model type and data type
model_type = "conv_lca"

#data_type = "cifar10"
data_type = "vanhateren"

## Import params
params, schedule = pp.get_params(model_type)
if "rand_seed" in params.keys():
  params["rand_state"] = np.random.RandomState(params["rand_seed"])
params["data_type"] = data_type

## Import data
data = ds.get_data(params)
params["input_shape"] = list(data["train"].images.shape[1:])
params["num_pixels"] = data["train"].num_pixels

## Import model
model = mp.get_model(model_type, params, schedule)

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  ## Need to provide shape if batch_size is used in graph
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros([params["batch_size"]]+model.input_shape, dtype=np.float32)})

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  if model.cp_load:
    if model.cp_load_step is None:
      cp_load_prefix = (model.cp_load_dir+model.cp_load_name+"_v"+model.cp_load_ver
        +"_weights")
      cp_load_file = tf.train.latest_checkpoint(cp_load_prefix)
    else:
      cp_load_file = (model.cp_load_dir+model.cp_load_name+"_v"+model.cp_load_ver
        +"_weights-"+str(model.cp_load_step))
    model.load_weights(sess, cp_load_file)

  avg_time = 0
  for sch_idx, sch in enumerate(schedule):
    model.sched_idx = sch_idx
    model.log_info("Beginning schedule "+str(sch_idx))
    for b_step in range(model.get_sched("num_batches")):
      data_batch = data["train"].next_batch(model.batch_size)
      input_data = data_batch[0][..., np.newaxis]
      input_labels = data_batch[1]

      ## Get feed dictionary for placeholders
      feed_dict = model.get_feed_dict(input_data, input_labels)

      ## Normalize weights
      if hasattr(model, "norm_weights"):
        if params["norm_weights"]:
          sess.run([model.norm_weights], feed_dict)

      # Reset activity from previous batch
      if hasattr(model, "reset_activity"):
        sess.run([model.reset_activity], feed_dict)

      ## Update weights
      batch_t0 = ti.time()
      for w_idx in range(len(model.get_sched("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)
      batch_t1 = ti.time()
      avg_time += (batch_t1-batch_t0)/model.batch_size

      ## Generate logs
      current_step = sess.run(model.global_step)
      if (current_step % model.log_int == 0
        and model.log_int > 0):
        model.print_update(input_data=input_data, input_labels=input_labels, batch_step=b_step+1)

      ## Plot weights & gradients
      if (current_step % model.gen_plot_int == 0
        and model.gen_plot_int > 0):
        model.generate_plots(input_data=input_data, input_labels=input_labels)

      ## Checkpoint
      if (current_step % model.cp_int == 0
        and model.cp_int > 0):
        save_dir = model.write_checkpoint(sess)
        if hasattr(model, "val_on_cp"):
          if model.val_on_cp: #Compute validation accuracy
            val_images = data["val"].images
            val_labels = data["val"].labels
            with tf.Session(graph=model.graph) as tmp_sess:
              val_feed_dict = model.get_feed_dict(val_images, val_labels)
              tmp_sess.run(model.init_op, val_feed_dict)
              model.weight_saver.restore(tmp_sess,
                save_dir+"_weights-"+str(current_step))
              if hasattr(model, "full_inference"):
                sess.run([model.full_inference], val_feed_dict)
              if hasattr(model, "step_inference"):
                for step in range(model.num_steps):
                  sess.run([model.step_inference], val_feed_dict)
              val_accuracy = (
                np.array(tmp_sess.run(model.accuracy, val_feed_dict)).tolist())
              stat_dict = {"validation_accuracy":val_accuracy}
              js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
              model.log_info("<stats>"+js_str+"</stats>")

  save_dir = model.write_checkpoint(sess)
  avg_time /= model.get_sched("num_batches")
  model.log_info("Avg time per image: "+str(avg_time))
  t1=ti.time()
  tot_time=float(t1-t0)
  out_str = ("Training on "+str(sch_idx*model.get_sched("num_batches")*model.params["batch_size"])
    +" Images is Complete. Total time was "+str(tot_time)+" seconds.\n")
  model.log_info(out_str)
