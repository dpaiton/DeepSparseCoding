import os
import sys
import argparse

import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from DeepSparseCoding.tf1x.utils.logger import Logger
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.data.data_selector as ds
import DeepSparseCoding.tf1x.models.model_picker as mp
import DeepSparseCoding.tf1x.analysis.analysis_picker as ap

class params(object):
  def __init__(self):
    self.device = '/gpu:0'
    #Which dataset to run analysis on, options are "train", "val", or "test"
    self.analysis_dataset = 'test'
    #Output directory file
    self.save_info = 'analysis_' + self.analysis_dataset
    # If false, append to log file
    self.overwrite_analysis_log = True
    # Load in training run stats from log file
    self.do_run_analysis = False
    # Evaluate model variables (specified in analysis class) on images
    self.do_evals = False
    # Dictionary fitting
    self.do_basis_analysis = False
    # LCA Inference analysis
    self.do_inference = False #TODO: Does not work for lca_subspace
    # Activity triggered averages
    self.do_atas = False #TODO: this can produce outputs that are too big for npz; need to batch?
    # Recon adversarial image analysis
    self.do_recon_adversaries = False # TODO: broken for rica
    #Classification adversarial image analysis
    self.do_class_adversaries = False
    # Find optimal stimulus using gradient methods
    self.do_neuron_visualization = False # adversaries must be False
    # Patchwise image recon
    self.do_full_recon = False
    # Orientation and Cross-Orientation analysis
    self.do_orientation_analysis = False # TODO: broken for ae_deep
    # Reconstructions from individual groups for subspace Sparse Coding
    self.do_group_recons = False
    # How many images to use for analysis, patches are generated from these
    self.num_analysis_images = 150#1000
    self.whiten_batch_size = 10 # for VH dataset
    # How many input patches to create - only used if model calls for patching
    self.num_patches = 10#e4
    # How many images to use in the ATA analysis
    # NOTE: No warning is given if this is greater than the number of available images
    self.num_ata_images = 5e2
    # How many noise images to compute noise ATAs
    self.num_noise_images = 1e3
    # How many random images to average over for inference statistics
    self.num_inference_images = 5
    # Number of images used for LCA_PCA cov matrix (should not be greater than num_patches)
    self.num_LCA_PCA_cov_images = int(1e5)
    # Edge size of full (square) image (for full_recon)
    self.image_edge_size = 128
    # Fourier analysis padding for weight fitting
    self.ft_padding = 128
    # How many inference steps to perform (None uses model params)
    self.num_inference_steps = None
    # Which dataset images to use for inference (None uses random)
    self.inference_img_indices = None
    #Adversarial params
    self.adversarial_num_steps = 500 # Step size for adversarial attacks
    self.adversarial_attack_method = 'carlini_targeted'#'kurakin_targeted'#"carlini_targeted"
    if self.do_class_adversaries or self.do_recon_adversaries:
        self.save_info += '_'+self.adversarial_attack_method # To avoid overwriting
    self.adversarial_step_size = 0.005 # learning rate for optimizer
    self.adversarial_max_change = None # maximum size of adversarial perturation
    self.carlini_change_variable = True
    self.adv_optimizer = 'sgd'
    self.adversarial_target_method = 'random' # Not used if attack_method is untargeted#TODO support specified
    self.adversarial_clip = True
    self.adversarial_clip_range = [0.0, 1.0] # Maximum range of image values
    self.carlini_recon_mult = [0.5]#list(np.arange(.5, 1, .1))
    self.adversarial_save_int = 1 # Interval at which to save adv examples to the npz file
    self.eval_batch_size = 50 # batch size for computing adv examples
    self.adversarial_input_id = None#list(range(100)) # Which adv images to use; None to use all
    self.confidence_threshold = 0.9
    self.temperature =1.0# 0.65#1.0
    #TODO
    #Parameter for "specified" target_method
    #Only for class attack
    #Need to be a list or numpy array of size [adv_batch_size]
    self.adversarial_target_labels = None
    # Rescale inputs to match dataset scales used during training
    self.input_scale = 1.0 # TODO: Get input_scale from log file
    # Which neurons to run tuning experiments on (None to do all)
    self.neuron_indices = None
    # Contrasts for orientation experiments
    self.contrasts = [0.5]
    # Phases for orientation experiments
    self.phases = np.linspace(-np.pi, np.pi, 8)
    # Orientations for orientation experiments
    self.orientations = np.linspace(0.0, np.pi, 16)
    # Optimal stimulus calculation
    self.neuron_vis_num_steps = int(4e5)
    self.neuron_vis_step_size = 2e-4
    self.neuron_vis_save_int = 1000
    self.neuron_vis_stim_save_int = int(1e5)
    self.neuron_vis_clip = False
    self.neuron_vis_clip_range = [1.0]
    self.neuron_vis_method = "erhan"
    self.neuron_vis_norm_magnitude = None
    self.neuron_vis_l2_regularize_coeff = 0.001
    self.neuron_vis_variation_coeff = 0.005
    self.neuron_vis_optimizer = "sgd"
    self.neuron_vis_target_layer = None
    # TODO: we are temporarily assigning a 1-hot vector for analysis, but we could pass a specific
    #   selection vector instead of a target_neuron_idx
    self.neuron_vis_targets = np.random.choice(range(10), 5, replace=False)
    self.neuron_vis_target_neuron_idx = self.neuron_vis_targets[0] # TODO: Clean this up...
    self.neuron_vis_selection_vector = np.zeros(10) # TODO: avoid hard-coding num neurons
    self.neuron_vis_selection_vector[self.neuron_vis_target_neuron_idx] = 1
    self.rand_seed = 123

parser = argparse.ArgumentParser()

# Get params, set dirs
analysis_params = params() # construct object
analysis_params.projects_dir = os.path.expanduser("~")+"/Work/Projects/"

# Load arguments
model_name_list = os.listdir(analysis_params.projects_dir)
parser.add_argument("model_name", help=", ".join(model_name_list))

parser.add_argument("model_version",
  help="Specify the string that was used for the 'version' parameter")

args = parser.parse_args()
analysis_params.model_name = args.model_name
analysis_params.version = args.model_version
analysis_params.model_dir = analysis_params.projects_dir+analysis_params.model_name

model_temperatures = {}
model_temperatures['mlp_768_mnist'] = 1.0
model_temperatures['slp_lca_768_latent_mnist'] = 0.65
model_temperatures['mlp_1568_mnist'] = 0.5
model_temperatures['slp_lca_1568_mnist'] = 0.68
model_temperatures['mlp_3layer_cosyne_mnist'] = 1.0
model_temperatures['mlp_lca_768_latent_75_steps_mnist'] = 0.75
model_temperatures['mlp_1568_3layer_mnist'] = 1.0
model_temperatures['mlp_lca_1568_latent_75_steps_mnist'] = 1.0
if analysis_params.model_name in model_temperatures.keys():
  analysis_params.temperature = model_temperatures[analysis_params.model_name]

model_log_file = (analysis_params.model_dir+"/logfiles/"+analysis_params.model_name
  +"_v"+analysis_params.version+".log")
model_logger = Logger(model_log_file, overwrite=False)
model_log_text = model_logger.load_file()
model_params = model_logger.read_params(model_log_text)[-1]
model_params.temperature = analysis_params.temperature
analysis_params.model_type = model_params.model_type

# Initialize & setup analyzer
analyzer = ap.get_analyzer(analysis_params.model_type)
analyzer.setup(analysis_params)

analysis_params.data_type = analyzer.model_params.data_type
if(analyzer.model_params.data_type == "vanhateren"):
  analyzer.model_params.whiten_batch_size = analysis_params.whiten_batch_size
analyzer.model_params.num_images = analysis_params.num_analysis_images
if hasattr(analyzer.model_params, "extract_patches") and analyzer.model_params.extract_patches:
  analyzer.model_params.num_patches = analysis_params.num_patches

data = ds.get_data(analyzer.model_params)

data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)
analyzer.model_params.data_shape = list(data["train"].shape[1:])

#analyzer.model_schedule[0]["sparse_mult"]  = 0.4
analyzer.setup_model(analyzer.model_params)
#analyzer.model_params.input_shape = [
#  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

analyzer.run_analysis(data[analysis_params.analysis_dataset].images,
  data[analysis_params.analysis_dataset].labels,
  save_info=analysis_params.save_info)

if analysis_params.do_full_recon:
  class img_params():
    data_type = analysis_params.data_type
    num_images = 2
    extract_patches = False
    image_edge_size = analysis_params.image_edge_size
    data_dir = os.path.expanduser("~")+"/Work/Datasets/"
    rand_seed = analysis_params.rand_seed
    rand_state = np.random.RandomState(analysis_params.rand_seed)
  full_img = dp.reshape_data(ds.get_data(img_params)["train"].images[0], flatten=False)[0]
  analyzer.run_patch_recon_analysis(full_img, save_info=analysis_params.save_info)
