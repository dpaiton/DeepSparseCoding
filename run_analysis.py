import matplotlib
matplotlib.use("Agg")

import os
import numpy as np

import analysis.analysis_picker as ap
import data.data_picker as dp
import utils.plot_functions as pf

## Parameters for analysis
analysis_params = {
  "model_type": "density_learner",
  "model_name": "density",
  "version": "0.0",
  #"batch_index": 500,
  "save_weights": True,
  "act_trig_avgs": True,
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

if analyzer.eval_train or analyzer.act_trig_avgs:
  analyzer.model_params["epoch_size"] = 100
  train_imgs = dp.get_data(analyzer.model_params["data_type"],
    analyzer.model_params)["train"].images
  train_model_outputs = analyzer.evaluate_model(train_imgs)
  if analyzer.save_weights:
    weight_list = [(key, val)
      for key, val in train_model_outputs.items()
      if "weights" in key.lower()]
    if len(weight_list) > 0:
      for weight_tuple in weight_list:
        #import IPython; IPython.embed(); raise SystemExit
        weight_name = weight_tuple[0].split("/")[-1].split(":")[0]
        if not os.path.exists(analyzer.out_dir+"/weights"):
          os.makedirs(analyzer.out_dir+"/weights")
        save_filename = analyzer.out_dir+"/weights/"+weight_name+".npz"
        np.savez(save_filename, data=weight_tuple[1])
    else:
      assert False, ("save_weights flag is True, but there are no weight"
        +"outputs from analyzer.evaluate_model")
  if analyzer.act_trig_avgs:
    ## Compute activity triggered averages on input data
    train_atas = analyzer.compute_atas(train_model_outputs["weights/phi:0"],
      train_model_outputs["inference/activity:0"], train_imgs)
    ata_filename = (analyzer.out_dir+"act_trig_avg_images_v"
      +analyzer.version+analyzer.file_ext)
    ata_title = "Activity triggered averages on image data"
    num_pixels, num_neurons = train_atas.shape
    pf.plot_data_tiled(train_atas.T.reshape(num_neurons,
      int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))), normalize=False,
      title=ata_title, vmin=np.min(train_atas), vmax=np.max(train_atas),
      save_filename=ata_filename)
    ## Compute activity triggered averages on noise data
    noise_data = np.random.standard_normal(train_imgs.shape)
    noise_model_outputs = analyzer.evaluate_model(noise_data)
    noise_atas = analyzer.compute_atas(noise_model_outputs["weights/phi:0"],
      noise_model_outputs["inference/activity:0"], noise_data)
    ata_filename = (analyzer.out_dir+"act_trig_avg_images_v"
      +analyzer.version+analyzer.file_ext)
    ata_title = "Activity triggered averages on image data"
    num_pixels, num_neurons = noise_atas.shape
    pf.plot_data_tiled(noise_atas.T.reshape(num_neurons,
      int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))), normalize=False,
      title=ata_title, vmin=np.min(noise_atas), vmax=np.max(noise_atas),
      save_filename=ata_filename)

#import IPython; IPython.embed(); raise SystemExit
