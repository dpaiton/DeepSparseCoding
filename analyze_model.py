import os
import numpy as np
import tensorflow as tf
import data.data_selector as ds
import utils.data_processing as dp
import analysis.analysis_picker as ap

analysis_params = {
  "model_type": "conv_lca",#"lca",
  "model_name": "conv_lca",#"lca",
  "version": "0.0",
  "save_info": "analysis",
  "device": "/gpu:0",
  "data_type": "vanHateren",#"vanHateren",
  "num_patches": 1e4,
  "overwrite_analysis_log": False,
  "do_basis_analysis": False, # Dictionary fitting
  "do_inference": False, # LCA Inference analysis
  "do_atas": False, # Activity triggered averages
  "do_adversaries": True, # Adversarial image analysis
  "do_full_recon": False, # Patchwise image recon
  "do_orientation_analysis": False, # Orientation and Cross-Orientation analysis
  "ft_padding": 32, # Fourier analysis padding for weight fitting
  "num_inference_images": 5, # How many random images to average over for inference statistics
  "inference_img_indices": None, # Which dataset images to use for inference
  "cov_num_images": int(1e5), # Number of images used to compute cov matrix (LCA_PCA)
  "num_noise_images": 300, # How many noise images to compute noise ATAs
  "adversarial_num_steps": 10, # Number of adversarial image updates
  "input_scale": 2.0,#13, # Will vary depending on preprocessing
  "neuron_indices": None, # Which neurons to run tuning experiments on (None to do all)
  "contrasts": [0.1, 0.2, 0.3, 0.4, 0.5], # Contrasts for orientation experiments
  "phases": np.linspace(-np.pi, np.pi, 8), # Phases for orientation experiments
  "orientations": np.linspace(0.0, np.pi, 16)} # Orientations for orientation experiments

# Computed params
analysis_params["model_dir"] = (os.path.expanduser("~")+"/Work/Projects/"
  +analysis_params["model_name"])
analyzer = ap.get_analyzer(analysis_params)

analyzer.model_params["data_type"] = analysis_params["data_type"]
analyzer.model_params["num_patches"] = analysis_params["num_patches"]

data = ds.get_data(analyzer.model_params)
data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)

analyzer.model_params["data_shape"] = list(data["train"].shape[1:])
analyzer.setup_model(analyzer.model_params, analyzer.model_schedule)
analyzer.model_params["input_shape"] = [
  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

analyzer.run_analysis(data["train"].images, save_info=analysis_params["save_info"])

if analysis_params["do_full_recon"]:
  img_params = {"data_type": analysis_params["data_type"], "num_images": 2, "extract_patches": False,
    "image_edge_size": 256, "data_dir": os.path.expanduser("~")+"/Work/Datasets/", "random_seed": 5}
  full_img = dp.reshape_data(ds.get_data(img_params)["train"].images[0], flatten=False)[0]
  analyzer.run_patch_recon_analysis(full_img, save_info=analysis_params["save_info"])
