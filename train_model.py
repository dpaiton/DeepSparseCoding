import matplotlib
matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds

import time as ti
t0 = ti.time()

## Specify model type and data type
#model_type = "mlp"
#model_type = "ica"
#model_type = "ica_pca"
model_type = "lca"
#model_type = "lca_pca"
#model_type = "lca_pca_fb"
#model_type = "conv_lca"
#model_type = "gradient_sc"
#model_type = "sigmoid_autoencoder"
#model_type = "density_learner"

#data_type = "cifar10"
#data_type = "mnist"
data_type = "vanhateren"
#data_type = "field"
#data_type = "synthetic"

## Import params
params, schedule = pp.get_params(model_type)
if "rand_seed" in params.keys():
  params["rand_state"] = np.random.RandomState(params["rand_seed"])
params["data_type"] = data_type

## Import data
data = ds.get_data(params)

## Import model
model = mp.get_model(model_type)
data = model.preprocess_dataset(data, params)
data = model.reshape_dataset(data, params)
params["data_shape"] = list(data["train"].shape[1:])
model.setup(params, schedule)
if "standardize_data" in params.keys() and params["standardize_data"]:
  model.log_info("Standardization was performed, mean was "+str(model.data_mean)
    +" and std was "+str(model.data_std))

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=model.graph) as sess:
  ## Need to provide shape if batch_size is used in graph
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros([params["batch_size"]]+params["data_shape"], dtype=np.float32)})

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  if model.cp_load:
    if model.cp_load_step is None:
      cp_load_file = tf.train.latest_checkpoint(model.cp_load_dir, model.cp_latest_filename)
    else:
      cp_load_file = (model.cp_load_dir+model.cp_load_name+"_v"+model.cp_load_ver
        +"_weights-"+str(model.cp_load_step))
    model.load_weights(sess, cp_load_file)

  avg_time = 0
  for sch_idx, sch in enumerate(schedule):
    model.sched_idx = sch_idx
    model.log_info("Beginning schedule "+str(sch_idx))
    for b_step in np.arange(model.get_schedule("num_batches")):
      data_batch = data["train"].next_batch(model.batch_size)
      input_data = data_batch[0]
      input_labels = data_batch[1]

      ## Get feed dictionary for placeholders
      feed_dict = model.get_feed_dict(input_data, input_labels)

      batch_t0 = ti.time()
      ## Update weights
      for w_idx in range(len(model.get_schedule("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)
      batch_t1 = ti.time()
      avg_time += (batch_t1-batch_t0)/model.batch_size

      ## Normalize weights
      if hasattr(model, "norm_weights"):
        if params["norm_weights"]:
          sess.run([model.norm_weights], feed_dict)

      ## Generate logs
      current_step = sess.run(model.global_step)
      if (current_step <= 1 and model.gen_plot_int > 0):
        model.print_update(input_data=input_data, input_labels=input_labels, batch_step=b_step+1)
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
              js_str = model.js_dumpstring(stat_dict)
              model.log_info("<stats>"+js_str+"</stats>")

  save_dir = model.write_checkpoint(sess)
  avg_time /= model.get_schedule("num_batches")
  model.log_info("Avg time per image: "+str(avg_time)+" seconds")
  t1=ti.time()
  tot_time=float(t1-t0)
  out_str = (
    "Training on "+str(sch_idx*model.get_schedule("num_batches")*model.params["batch_size"])
    +" Images is Complete. Total time was "+str(tot_time)+" seconds.\n")
  model.log_info(out_str)
  print("Training Complete\n")
