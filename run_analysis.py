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
  "model_name": "kl_run",
  "out_dir": os.path.expanduser("~")+"/Work/Analysis/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  #"batch_index": 500,
  "eval_train": True,
  "eval_test": True,
  "eval_val": True,
  "file_ext": ".pdf",
  "device": "/cpu:0",
  # K&L specific params
  "eval_inference": True,
  "eval_density_weights": True}

model_dir = (os.path.expanduser("~")+"/Work/Projects/"
  +analysis_params["model_name"])

log_file = (model_dir+"/logfiles/"
  +analysis_params["model_name"]+"_v"+analysis_params["version"]+".log")

log_text = log_parser.load_file(log_file)

model_params = log_parser.read_params(log_text)
model_params["out_dir"] = analysis_params["out_dir"]
model_params["data_dir"] = analysis_params["data_dir"]

np_rand_state = np.random.RandomState(model_params["rand_seed"])

model_schedule = log_parser.read_schedule(log_text)

if "batch_index" in  analysis_params:
  assert analysis_params["batch_index"] > 0
else:
  batch_idx = 0
  for schedule in model_schedule:
    batch_idx += schedule["num_batches"]
  analysis_params["batch_index"] = batch_idx

analyzer = kl_analysis(analysis_params)

model_stats = log_parser.read_stats(log_text)
stats_fig = analyzer.plot_stats(model_stats)

cp_loc = (model_dir+"/checkpoints/"+analysis_params["model_name"]+"_v"
  +analysis_params["version"]+"_full-"+str(analysis_params["batch_index"]))

model_params["log_to_file"] = False
model_params["out_dir"] = analysis_params["out_dir"]
model = mp.get_model(model_params, model_schedule)
model.log_info("Analysis params:\n%s\n"%(str(analysis_params)))

data = load_MNIST(model_params["data_dir"],
  normalize_imgs=model_params["norm_images"],
  rand_state=np_rand_state)["test"]

data_model_states = analyzer.evaluate_model(model, data, cp_loc)
analyzer.plot_ata(data_model_states["ata"], "MNIST")

noise_images = np.random.randn(data.images.shape[0], data.images.shape[1])
noise_data = type('', (), {})() # empty object
noise_data.images = noise_images
noise_model_states = analyzer.evaluate_model(model, noise_data, cp_loc)
analyzer.plot_ata(noise_model_states["ata"], "NOISE")

import IPython; IPython.embed(); raise SystemExit

