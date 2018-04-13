import os
import numpy as np
import tensorflow as tf
import data.data_selector as ds
import analysis.analysis_picker as ap

analysis_params = {
  #"model_type": "ica",
  #"model_type": "lca",
  #"model_type": "conv_lca",
  #"model_type": "sparse_autoencoder",
  "model_type": "gdn_autoencoder",
  #"model_name": "ica",
  #"model_name": "lca_vh_ft_1c_ht",
  #"model_name": "lca_pca_512_vh_ft_white",
  #"model_name": "conv_lca_vh",
  #"model_name": "lca_vh_ft_64d_96n",
  "model_name": "gdn_autoencoder",
  #"version": "5.0", #lca
  #"version": "0.0", #lca, sparse_autoencoder
  #"version": "1.0", #ICA
  "version": "1.0", #gdn_autoencoder
  "data_type": "vanHateren",
  "device": "/gpu:0",
  #"save_info": "full_imgs",
  "save_info": "analysis",
  "num_patches": 1e4,
  "ft_padding": 16, #Fourier analysis padding for weight fitting
  "num_inference_images": 5, #How many random images to average over for inference statistics
  "cov_num_images": int(3e5), #number of images used to compute cov matrix (LCA_PCA)
  "neuron_indices": None,
  "contrasts": [0.1, 0.2, 0.3, 0.4, 0.5],
  "phases": np.linspace(-np.pi, np.pi, 12),
  "orientations": np.linspace(0.0, np.pi, 32)}

# Computed params
#analysis_params["model_dir"] = (os.path.expanduser("~")+"/Work/Projects/nowht_v3/"
#  +analysis_params["model_name"]+"_v"+analysis_params["version"])
analysis_params["model_dir"] = (os.path.expanduser("~")+"/Work/Projects/"
  +analysis_params["model_name"])
analyzer = ap.get_analyzer(analysis_params)

analyzer.model_params["data_type"] = analysis_params["data_type"]
analyzer.model_params["num_patches"] = analysis_params["num_patches"]
data = ds.get_data(analyzer.model_params)
data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)
analyzer.model_params["data_shape"] = list(data["train"].shape[1:])
analyzer.model.setup(analyzer.model_params, analyzer.model_schedule)
analyzer.model_params["input_shape"] = [
  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

analyzer.run_analysis(data["train"].images, save_info=analysis_params["save_info"])
