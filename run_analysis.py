import matplotlib
matplotlib.use("Agg")

import os
import tensorflow as tf
import numpy as np

from data.MNIST import load_MNIST
import models.model_picker as mp
from analysis.karklin_lewicki import karklin_lewicki as kl_analysis
import utils.log_parser as log_parser

analysis_params = {
  "version": "0.0",
  "model_name": "test",
  "out_dir": os.path.expanduser("~")+"/Work/Analysis/",
  "batch_index": -1,
  "eval_train": True,
  "eval_test": True,
  "eval_val": True,
  "file_ext": ".pdf",
  "device": "/cpu:0",
  # K&L specific params
  "eval_inference": True,
  "eval_density_weights": True}

log_file_loc = (os.path.expanduser("~")
  +"/Work/Projects/"+analysis_params["model_name"]+"/logfiles/"
  +analysis_params["model_name"]+"_v"+analysis_param["version"]+".log")

log_text = log_parser.load_file(log_file_loc)
model_params = log_parser.read_params(log_text)
model_schedule = log_parser.read_schedule(log_text)
model_stats = log_parser.read_stats(log_text)

analyzer = kl_analysis(analysis_params)
stats_fig = analyzer.plot_stats(model_stats)

np_rand_state = np.random.RandomState(model_params["rand_seed"])

data = load_MNIST(model_params["data_dir"],
  normalize_imgs=model_params["norm_images"],
  rand_state=np_rand_state)

model_params["log_to_file"] = False
model_params["out_dir"] = analysis_params["out_dir"]
model = mp.get_model(model_params, model_schedule)

with tf.Session(graph=model.graph) as sess:
  model.sched_idx = 0
  sess.run(model.init_op,
    feed_dict={model.x:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32)})

  mnist_batch = data["train"].next_batch(model.batch_size)
  input_images = mnist_batch[0].T

  feed_dict = model.get_feed_dict(input_images)

  ##TODO: Move this to karklin_lewicki analysis
  _, u_t, v_t, = sess.run([model.do_inference,
    model.u_t, model.v_t,],
    feed_dict)
  print(np.max(u_t))
  print("\n")
  print(np.max(v_t))
  print("\n")

import IPython; IPython.embed(); raise SystemExit

