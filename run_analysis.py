import matplotlib
matplotlib.use("Agg")

import os
import numpy as np

import analysis.analysis_picker as ap
import data.data_picker as dp

## Parameters for analysis
analysis_params = {
  "model_type": "lca",
  "model_name": "pretrain",
  "version": "2.0",
  #"batch_index": 500,
  "eval_train": True,
  "eval_test": False,
  "eval_val": False,
  "file_ext": ".pdf",
  "device": "/cpu:0",
  "eval_inference": True,
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/"}
analysis_params["model_dir"] = (os.path.expanduser("~")+"/Work/Projects/"
  +analysis_params["model_name"])

## Get analyzer object
analyzer = ap.get_analyzer(analysis_params)

## Generate universal plots
analyzer.save_log_stats()

if analysis_params["eval_train"]:
  analyzer.model_params["epoch_size"] = 100
  train_imgs = dp.get_data(analyzer.model_params["data_type"],
    analyzer.model_params)["train"].images
  model_train_output = analyzer.evaluate_model(train_imgs)
  atas = analyzer.compute_atas(model_train_output["weights/phi:0"],
    model_train_output["inference/activity:0"], train_imgs)

#model = mp.get_model(model_params["model_type"], model_params, model_schedule)
#model.log_info("Analysis params:\n%s\n"%(str(analysis_params)))
#
#data = load_MNIST(model_params["data_dir"],
#  normalize_imgs=model_params["norm_images"],
#  rand_state=np_rand_state)["test"]
#
#data_model_states = analyzer.evaluate_model(model, data,
#  analysis_params["cp_loc"])
#analyzer.save_ata(data_model_states["ata"], "MNIST")
#
#noise_images = np.random.randn(data.images.shape[0], data.images.shape[1])
#noise_data = type('', (), {})() # empty object
#noise_data.images = noise_images
#noise_model_states = analyzer.evaluate_model(model, noise_data,
#  analysis_params["cp_loc"])
#analyzer.save_ata(noise_model_states["ata"], "NOISE")
#
#import IPython; IPython.embed(); raise SystemExit
