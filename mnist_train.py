import os
import numpy as np
import tensorflow as tf
import models.model_picker as mp
from data.MNIST import load_MNIST

params = {
  "model_type": "karklin_lewicki",
  "model_name": "test",
  "output_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  #"rectify_a": True,
  "rectify_u": True,
  "rectify_v": False,
  "norm_images": False,
  "norm_a": False,
  "norm_weights": False,
  "batch_size": 100,
  "num_pixels": 784,
  #"num_neurons": 400,
  "num_u": 200,
  "num_v": 20,
  #"num_val": 0,
  #"num_labeled": 60000,
  "num_steps": 20,
  "u_step_size": 0.1,
  "v_step_size": 0.05,
  #TODO: The files don't stick around? Only ever have last few.
  # max to keep: https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
  # set as param? Or just restructure checkpointing.
  "cp_int": 15000,
  #"val_on_cp": True,
  "cp_load": True,
  "cp_load_name": "pretrain",
  "cp_load_val": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["phi", "bias", "w"],
  "stats_display": 10,
  "generate_plots": 50000,
  "display_plots": False,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890}

schedule = [
  {"weights": ["a", "b"],
  "recon_mult": 1.0,
  "sparse_mult": 0.10,
  "weight_lr": [0.1, 0.1],
  "decay_steps": [30000]*2,
  "decay_rate": [0.8]*2,
  "staircase": [True]*2,
  "num_batches": 100}]

## Get data
np_rand_state = np.random.RandomState(params["rand_seed"])
data = load_MNIST(params["data_dir"],
  normalize_imgs=params["norm_images"],
  rand_state=np_rand_state)

model = mp.get_model(params, schedule)

model.log_info("This is a test of the new model code")

model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32)})

  model.write_graph(sess.graph_def)

  for sch_idx, sch in enumerate(schedule):
    model.sched_idx = sch_idx
    for b_step in range(model.get_sched("num_batches")):
      mnist_batch = data["train"].next_batch(model.batch_size)
      input_images = mnist_batch[0].T

      feed_dict = model.get_feed_dict(input_images)

      ## Normalize weights
      if params["norm_weights"]:
        sess.run(model.normalize_weights)

      ## Update weights
      for w_idx in range(len(model.get_sched("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)

      ## Generate outputs
      current_step = sess.run(model.global_step)
      if (current_step % model.stats_display == 0
        and model.stats_display > 0):
        model.print_update(input_images, None, b_step+1)

  import IPython; IPython.embed()
