import os
import numpy as np
import argparse
import tensorflow as tf
from utils.logger import Logger
import utils.data_processing as dp
import data.data_selector as ds
import models.model_picker as mp
import analysis.analysis_picker as ap

class params(object):
  def __init__(self):
    self.device = "/gpu:0"
    #Which dataset to run analysis on, options are "train", "val", or "test"
    self.analysis_dataset = "test"
    #Output directory file
    self.save_info = "analysis_" + self.analysis_dataset
    # If false, append to log file
    self.overwrite_analysis_log = True
    # Load in training run stats from log file
    self.do_run_analysis = False
    # Evaluate model variables (specified in analysis class) on images
    self.do_evals = False
    # Dictionary fitting
    self.do_basis_analysis = False
    # LCA Inference analysis
    self.do_inference = False
    # Activity triggered averages
    self.do_atas = False
    # Recon adversarial image analysis
    self.do_recon_adversaries = False # TODO: broken for rica
    #Classification adversarial image analysis
    self.do_class_adversaries = True
    # Patchwise image recon
    self.do_full_recon = False
    # Orientation and Cross-Orientation analysis
    self.do_orientation_analysis = False
    # How many images to use for analysis, patches are generated from these
    self.num_analysis_images = 1000
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

    #Adversarial params
    #self.adversarial_num_steps = 500 # Step size for adversarial attacks
    #self.adversarial_attack_method = "kurakin_untargeted" # Only for class attack

    #self.adversarial_attack_method = "kurakin_targeted"
    #self.carlini_recon_mult = [.5]

    self.adversarial_attack_method = "carlini_targeted"
    self.carlini_recon_mult = [.5]
    self.adversarial_num_steps = 1000
    #self.carlini_recon_mult = list(np.arange(.1, 1, .1))

    #To avoid overwriting
    self.save_info += "_"+self.adversarial_attack_method

    self.adversarial_step_size = 0.001
    #self.adversarial_max_change = None #0.03 #0.3 #For cifar10/mnist
    self.adversarial_max_change = 0.03 #0.3 #For cifar10/mnist
    #TODO support specified
    self.adversarial_target_method = "random" #Not used if attach_method is untargeted
    self.adversarial_clip = True
    self.adversarial_clip_range = [0.0, 1.0]


    #Interval at which to save adversarial examples to the npy file
    #self.adversarial_save_int = 1
    self.adversarial_save_int = 100

    self.eval_batch_size = 10

    #Specify which adv to use here
    #If none, use all
    self.adversarial_input_id = list(range(100))

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
model_name_list = os.listdir(analysis_params.projects_dir)
parser.add_argument("model_name", help=", ".join(model_name_list))

parser.add_argument("model_version",
  help="Specify the string that was used for the 'version' parameter")

args = parser.parse_args()
analysis_params.model_name = args.model_name
analysis_params.version = args.model_version
analysis_params.model_dir = analysis_params.projects_dir+analysis_params.model_name

model_log_file = (analysis_params.model_dir+"/logfiles/"+analysis_params.model_name
  +"_v"+analysis_params.version+".log")
model_logger = Logger(model_log_file, overwrite=False)
model_log_text = model_logger.load_file()
model_params = model_logger.read_params(model_log_text)[-1]
analysis_params.model_type = model_params.model_type

# Initialize & setup analyzer
analyzer = ap.get_analyzer(analysis_params.model_type)
analyzer.setup(analysis_params)

analysis_params.data_type = analyzer.model_params.data_type
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
