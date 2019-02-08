import matplotlib
matplotlib.use("Agg")

import time as ti
import numpy as np
import argparse
import tensorflow as tf
import params.param_picker as pp
import models.model_picker as mp
import data.data_selector as ds

#from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

model_list = mp.get_model_list()
parser.add_argument("model_type", help=", ".join(model_list))

data_list = ds.get_dataset_list()
parser.add_argument("data_type", help=", ".join(data_list))

args = parser.parse_args()
model_type = args.model_type
data_type = args.data_type

t0 = ti.time()

## Import params
params = pp.get_params(model_type)
params.set_data_params(data_type)

## Import data
data = ds.get_data(params)
if data_type.lower() == "field":
  params.whiten_data = False

## Import model
model = mp.get_model(model_type)
data = model.preprocess_dataset(data, params)
data = model.reshape_dataset(data, params)
params.data_shape = list(data["train"].shape[1:])
model.setup(params)

if hasattr(params, "standardize_data") and params.standardize_data:
  model.log_info("Standardization was performed, dataset mean was "+str(np.mean(model.data_mean))
    +" and std was "+str(np.mean(model.data_std)))
if hasattr(params, "norm_data") and params.norm_data:
  model.log_info("Normalization was performed by dividing the dataset by max(abs(data)), "
    +"max was "+str(model.data_max))

## Write model weight savers for checkpointing and visualizing graph
model.write_saver_defs()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=model.graph) as sess:
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  #Need the shape of input placeholders for running init
  init_feed_dict = {model.input_placeholder: np.zeros([params.batch_size,] + params.data_shape)}
  sess.run(model.init_op, init_feed_dict)

  sess.graph.finalize() # Graph is read-only after this statement
  model.write_graph(sess.graph_def)

  if params.cp_load:
    if params.cp_load_step is None:
      cp_load_file = tf.train.latest_checkpoint(params.cp_load_dir, params.cp_load_latest_filename)
    else:
      cp_load_file = (params.cp_load_dir+params.cp_load_name+"_v"+params.cp_load_ver
        +"-"+str(params.cp_load_step))
    model.load_checkpoint(sess, cp_load_file)

  avg_time = 0
  tot_num_batches = 0
  init = True
  for sch_idx, sch in enumerate(params.schedule):
    tot_num_batches += sch["num_batches"]
    model.sched_idx = sch_idx
    model.log_info("Beginning schedule "+str(sch_idx))
    for b_step in np.arange(sch["num_batches"]):
      data_batch = data["train"].next_batch(params.batch_size)
      input_data = data_batch[0]
      input_labels = data_batch[1]

      ## Get feed dictionary for placeholders
      feed_dict = model.get_feed_dict(input_data, input_labels)
      feed_dict = model.modify_input(feed_dict)

      batch_t0 = ti.time()

      ## Generate initial logs & plots
      if init:
        model.print_update(input_data=input_data, input_labels=input_labels, batch_step=b_step+1)
        model.generate_plots(input_data=input_data, input_labels=input_labels)
        init = False

      ## Update model weights
      sess_run_list = []
      for w_idx in range(len(model.get_schedule("weights"))):
        sess_run_list.append(model.apply_grads[sch_idx][w_idx])
      sess.run(sess_run_list, feed_dict)

      if model.params.optimizer == "lbfgsb":
        model.minimizer.minimize(session=sess, feed_dict=feed_dict)

      ## Normalize weights
      if params.norm_weights:
        sess.run([model.norm_weights], feed_dict)

      batch_t1 = ti.time()
      avg_time += (batch_t1-batch_t0)/params.batch_size

      ## Generate logs
      current_step = sess.run(model.global_step)
      if (current_step % params.log_int == 0 and params.log_int > 0):
        model.print_update(input_data=input_data, input_labels=input_labels, batch_step=b_step+1)

      ## Generate plots
      if (current_step % params.gen_plot_int == 0 and params.gen_plot_int > 0):
        model.generate_plots(input_data=input_data, input_labels=input_labels)

      ## Checkpoint & validate
      if (current_step % params.cp_int == 0
        and params.cp_int > 0):
        save_dir = model.write_checkpoint(sess)
        if params.val_on_cp: #Compute validation accuracy
          val_images = data["val"].images
          val_labels = data["val"].labels

          est_labels = model.evaluate_model_batch(params.eval_batch_size,
            val_images, labels = val_labels, var_nodes=[model.label_est])[model.label_est]

          val_accuracy = np.mean(np.argmax(val_labels, -1) == np.argmax(est_labels, -1))
          stat_dict = {"validation_accuracy":val_accuracy}
          js_str = model.js_dumpstring(stat_dict)
          model.log_info("<stats>"+js_str+"</stats>")

  #If training ends right after a validation, need to remodify inputs
  #for adv examples
  #TODO is there a better way to do this?
  if(current_step % params.cp_int == 0
    and params.cp_int > 0):
    feed_dict = model.modify_input(feed_dict)

  model.print_update(input_data=input_data, input_labels=input_labels, batch_step=b_step+1)
  model.generate_plots(input_data=input_data, input_labels=input_labels)
  save_dir = model.write_checkpoint(sess)
  avg_time /= tot_num_batches
  model.log_info("Avg time per image: "+str(avg_time)+" seconds")
  t1=ti.time()
  tot_time=float(t1-t0)
  out_str = (
    "Training on "+str(sch_idx*tot_num_batches*params.batch_size)
    +" Images is Complete. Total time was "+str(tot_time)+" seconds.\n")
  model.log_info(out_str)
  print("Training Complete\n")
