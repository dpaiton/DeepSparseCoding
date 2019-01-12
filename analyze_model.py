import os
import numpy as np
import argparse
import tensorflow as tf
import utils.data_processing as dp
import data.data_selector as ds
import models.model_picker as mp
import analysis.analysis_picker as ap

class params(object):
  def __init__(self):
    self.save_info = "analysis"
    self.device = "/gpu:0"
    # If false, append to log file
    self.overwrite_analysis_log = True
    # Load in training run stats from log file
    self.do_run_analysis = True
    # Evaluate model variables (specified in analysis class) on images
    self.do_evals = False
    # Dictionary fitting
    self.do_basis_analysis = False
    # LCA Inference analysis
    self.do_inference = False
    # Activity triggered averages
    self.do_atas = False
    # Recon adversarial image analysis
    self.do_recon_adversaries = True
    #Classification adversarial image analysis
    self.do_class_adversaries = False
    # Patchwise image recon
    self.do_full_recon = False
    # Orientation and Cross-Orientation analysis
    self.do_orientation_analysis = False
    # How many images to use for analysis, patches are generated from these
    self.num_analysis_images = 200
    # How many input patches to create - only used if model calls for patching
    self.num_patches = 1e4
    # How many images to use in the ATA analysis
    # NOTE: No warning is given if this is greater than the number of available images
    self.num_ata_images = 1e3
    # How many noise images to compute noise ATAs
    self.num_noise_images = 1e3
    # How many random images to average over for inference statistics
    self.num_inference_images = 5
    # Number of images used for LCA_PCA cov matrix (should not be greater than num_patches)
    self.num_LCA_PCA_cov_images = int(1e5)
    # Edge size of full (square) image (for full_recon)
    self.image_edge_size = 128
    # Fourier analysis padding for weight fitting
    self.ft_padding = 32
    # How many inference steps to perform (None uses model params)
    self.num_inference_steps = None
    # Which dataset images to use for inference (None uses random)
    self.inference_img_indices = None
    # Number of adversarial image updates
    self.adversarial_num_steps = 1000
    # Step size for adversarial attacks
    self.adversarial_step_size = 0.005
    #Attack method for adversarial attack, kurakin or carlini
    self.adversarial_attack_method = "carlini"
    #self.adversarial_attack_method = "kurakin"
    #Flag to define if adversarial example can go beyond image range
    self.adversarial_clip = True
    #Recon_mult tradeoff for carlini attack method
    #Can be a list to sweep
    self.recon_mult = list(np.arange(.1, 10, .1))
    #Target class for class adversary analysis
    #None for untargeted
    self.adversarial_target_label = None
    # Will vary depending on preprocessing
    self.input_scale = 1.0
    # Which neurons to run tuning experiments on (None to do all)
    self.neuron_indices = None
    # Contrasts for orientation experiments
    self.contrasts = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Phases for orientation experiments
    self.phases = np.linspace(-np.pi, np.pi, 8)
    # Orientations for orientation experiments
    self.orientations = np.linspace(0.0, np.pi, 16)

parser = argparse.ArgumentParser()

# Get params, set dirs
analysis_params = params() # construct object
analysis_params.projects_dir = os.path.expanduser("~")+"/Work/Projects/"

# Load arguments
model_type_list = mp.get_model_list()
parser.add_argument("model_type", help=", ".join(model_type_list))

model_name_list = os.listdir(analysis_params.projects_dir)
parser.add_argument("model_name", help=", ".join(model_name_list))

parser.add_argument("model_version",
  help="Specify the string that was used for the 'version' parameter")

args = parser.parse_args()
analysis_params.model_type = args.model_type
analysis_params.model_name = args.model_name
analysis_params.version = args.model_version
analysis_params.model_dir = analysis_params.projects_dir+analysis_params.model_name

# Initialize & setup analyzer
analyzer = ap.get_analyzer(analysis_params.model_type)
analyzer.setup(analysis_params)

analysis_params.data_type = analyzer.model_params.data_type
analyzer.model_params.num_images = analysis_params.num_analysis_images
if hasattr(analyzer.model_params, "extract_patches") and analyzer.model_params.extract_patches:
  analyzer.model_params.num_patches = analysis_params.num_patches

# Load data for analysis
data = ds.get_data(analyzer.model_params)
data = analyzer.model.preprocess_dataset(data, analyzer.model_params)
data = analyzer.model.reshape_dataset(data, analyzer.model_params)
analyzer.model_params.data_shape = list(data["train"].shape[1:])

#analyzer.model_schedule[0]["sparse_mult"]  = 0.4
analyzer.setup_model(analyzer.model_params)
#analyzer.model_params.input_shape = [
#  data["train"].num_rows*data["train"].num_cols*data["train"].num_channels]

analyzer.run_analysis(data["train"].images, data["train"].labels,
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
