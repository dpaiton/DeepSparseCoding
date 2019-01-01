import os
import numpy as np
import tensorflow as tf
import data.data_selector as ds
import utils.data_processing as dp
import analysis.analysis_picker as ap

class params(object):
  def __init__(self):
    self.model_type = "mlp"
    self.model_name = "mlp_mnist"
    self.version = "0.0"
    self.save_info = "analysis"
    self.device = "/gpu:0"
    self.num_patches = 1e4 # How many input patches to create - only used if model calls for patching
    self.overwrite_analysis_log = True # If false, append to log file
    self.do_basis_analysis = True # Dictionary fitting
    self.do_inference = True # LCA Inference analysis
    self.do_atas = False # Activity triggered averages
    self.do_adversaries = True # Adversarial image analysis
    self.do_full_recon = False # Patchwise image recon
    self.do_orientation_analysis = False # Orientation and Cross-Orientation analysis
    self.ft_padding = 32 # Fourier analysis padding for weight fitting
    self.num_inference_images = 5 # How many random images to average over for inference statistics
    self.num_inference_steps = None # How many inference steps to perform
    self.inference_img_indices = None # Which dataset images to use for inference
    self.cov_num_images = int(1e5) # Number of images used to compute cov matrix (LCA_PCA)
    self.num_noise_images = 300 # How many noise images to compute noise ATAs
    self.adversarial_num_steps = 1000 # Number of adversarial image updates
    self.adversarial_eps = 0.005 # Step size for adversarial attacks
    self.input_scale = 1.0 # Will vary depending on preprocessing
    self.neuron_indices = None # Which neurons to run tuning experiments on (None to do all)
    self.contrasts = [0.1, 0.2, 0.3, 0.4, 0.5] # Contrasts for orientation experiments
    self.phases = np.linspace(-np.pi, np.pi, 8) # Phases for orientation experiments
    self.orientations = np.linspace(0.0, np.pi, 16) # Orientations for orientation experiments

# Computed params
analysis_params = params() # construct object
analysis_params.model_dir = (os.path.expanduser("~")+"/Work/Projects/"+analysis_params.model_name)
analyzer = ap.get_analyzer(analysis_params)

analysis_params.data_type = analyzer.model_params.data_type
if hasattr(analyzer.model_params, "extract_patches") and analyzer.model_params.extract_patches:
  analyzer.model_params.num_patches = analysis_params.num_patches

data = ds.get_data(analyzer.model_params)
data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)

analyzer.model_params.data_shape = list(data["train"].shape[1:])
#analyzer.model_schedule[0]["sparse_mult"]  = 0.4
analyzer.setup_model(analyzer.model_params, analyzer.model_schedule)
analyzer.model_params.input_shape = [
  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

#import IPython; IPython.embed(); raise SystemExit
analyzer.run_analysis(data["train"].images, save_info=analysis_params.save_info)

if analysis_params.do_full_recon:
  img_params = {"data_type": analysis_params.data_type, "num_images": 2, "extract_patches": False,
    "image_edge_size": 256, "data_dir": os.path.expanduser("~")+"/Work/Datasets/", "random_seed": 5}
  full_img = dp.reshape_data(ds.get_data(img_params)["train"].images[0], flatten=False)[0]
  analyzer.run_patch_recon_analysis(full_img, save_info=analysis_params.save_info)
