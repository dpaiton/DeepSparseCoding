import os
import numpy as np
import tensorflow as tf
import models.model_picker as mp



params = {
  "model_type": "mlp",
  "model_name": "test",
  "output_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "auto_diff_u": True,
  "rectify_a": True,
  "norm_images": False,
  "norm_a": False,
  "norm_weights": True,
  "one_hot_labels": True,
  "batch_size": 100,
  "num_pixels": 784,
  "num_neurons": 400,
  "num_classes": 10,
  "num_val": 10000,
  "num_labeled": 50000, #[50000:1.0, 100:0.002, 20:0.0004]
  "dt": 0.001,
  "tau": 0.03,
  #TODO: The files don't stick around? Only ever have last few.
  # max to keep: https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
  # set as param? Or just restructure checkpointing.
  "cp_int": 15000,
  "val_on_cp": True,
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
  {"weights": ["phi"], # Pretrain
  "recon_mult": 1.0,
  "sparse_mult": 0.20,
  "sup_mult": 0.0,
  "fb_mult": 0.0,
  "ent_mult": 0.0,
  "num_steps": 30,
  "weight_lr": [0.15],
  "decay_steps": [30000],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 150000}]

## Get data
np_rand_state = np.random.RandomState(params["rand_seed"])
data = load_MNIST(


model = mp.get_model(params, schedule)

model.log_info("This is a test of the new model code")

model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  sess.run(model.init_op,
    feed_dict={model.s:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32), model.y:np.zeros((model.num_classes, model.batch_size),
    dtype=np.float32)})

  model.write_graph(sess.graph_def)

  model.sched_idx = 0





import IPython; IPython.embed()
