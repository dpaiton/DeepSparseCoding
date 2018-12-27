import os
import numpy as np
from params.base_params import Base_Params

class params(Base_Params):
  def __init__(self):
    """
    Additional modifiable parameters:
      num_preproc_threads [int] number of threads to use for data preprocessing
      downsample_images   [bool] if set, then downsample input images as a preprocessing step
      mle_step_size       [float] size of maximum likelihood estimator steps
      num_mle_steps       [int] number of maximum likelihood estimation steps for the entropy estimator
      num_triangles       [int] number of triangle kernels to use for the entropy estimator
      sigmoid_beta        [float] slope parameter for sigmoid activation function
      im_size_y           [int] number of pixel rows in an image matrix
      im_size_x           [int] number of pixel columns in an image matrix
      num_colors          [int] number of colors in the input
      patch_size_y        [list] where each element is the number of rows in the patch for the layer
      patch_size_x        [list] where each element is the number of columns in the patch for the layer
      input_channels      [list] where each element is the number of input channels for the layer
      output_channels     [list] where each element is the number of output channels for the layer
      w_strides           [list] where each element is the convolutional stride for the layer
      gdn_w_init_const    [float] initial value for the GDN weights
      gdn_b_init_const    [float] initial value for the GDN biases
      gdn_w_thresh_min    [float] minimum allowable value for GDN weights
      gdn_b_thresh_min    [float] minimum allowable value for GDN biases
      gdn_eps             [float] epsilon that will be added to the GDN denominator
      nmem                [int] number of memory devices in the latent space
      memristor_type      [str] type of memristor, can be "rram", "pcm", "gauss", or None
      memristor_data_loc  [str] location of memristor data for modeling the transfer function
    """
    super(params, self).__init__()
    self.model_type = "conv_gdn_autoencoder"
    self.model_name = "test_conv_gdn_autoencoder_ent"
    self.version = "0.0"
    self.vectorize_data = False
    self.norm_data = False
    self.center_data = False
    self.standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.whiten_method = "FT"
    self.lpf_data = True
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.im_size_y = 64
    self.im_size_x = 64
    self.batch_size = 50
    self.num_colors = 1
    self.downsample_images = True
    self.num_preproc_threads = 8
    self.mem_v_min = -1.0
    self.mem_v_max = 1.0
    self.sigmoid_beta = 1.0
    self.mle_step_size = 0.01
    self.num_mle_steps = 15
    self.num_triangles = 20
    self.n_mem = 448
    self.input_channels = [1, 128, 64]
    self.output_channels = [128, 64, 28]
    self.patch_size_y = [8, 9, 4]
    self.patch_size_x = [8, 9, 4]
    self.w_strides = [4, 2, 2]
    self.gdn_w_init_const = 0.1
    self.gdn_b_init_const = 0.1
    self.gdn_w_thresh_min = 1e-6
    self.gdn_b_thresh_min = 1e-6
    self.gdn_eps = 1e-6
    self.memristor_type = "rram"
    self.memristor_data_loc = os.path.expanduser("~")+"/CAE_Project/CAEs/data/Partial_Reset_PCM.pkl"
    self.optimizer = "annealed_sgd"
    self.cp_int = 100000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.cp_load_name = "conv_gdn_autoencoder_dropout"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 100000
    self.save_plots = True
    self.data_file ="/home/dpaiton/Work/Datasets/verified_images.txt"

    self.w_list = ["w"+str(idx) for idx in range(2*len(self.input_channels))]
    self.b_list = ["b"+str(idx) for idx in range(2*len(self.input_channels))]
    self.w_gdn_list = ["w_gdn"+str(idx) for idx in range(len(self.input_channels))]
    self.b_gdn_list = ["b_gdn"+str(idx) for idx in range(len(self.input_channels))]
    # Don't do igdn on last layer (reconstruction)
    self.w_igdn_list = ["w_igdn"+str(idx)
      for idx in range(len(self.input_channels), 2*len(self.input_channels)-1)]
    self.b_igdn_list = ["b_igdn"+str(idx)
      for idx in range(len(self.input_channels), 2*len(self.input_channels)-1)]
    
    self.conv_list = self.w_list + self.b_list
    self.gdn_list = self.w_gdn_list + self.b_gdn_list + self.w_igdn_list + self.b_igdn_list
    self.train_list = self.conv_list +  self.gdn_list
    self.cp_load_var = self.train_list
    
    weight_lr = [3.0e-4 for _ in range(len(self.conv_list))]
    weight_lr += [1.0e-4 for _ in range(len(self.gdn_list))]
    decay_rate = [0.8 for _ in range(len(self.train_list))]
    staircase = [True for _ in range(len(self.train_list))]
    
    self.schedule = [
      {"weights": self.train_list,
      "ent_mult": 0.1,
      "ramp_slope": 1.0,
      "decay_mult": 0.002,
      "noise_var_mult": 0.0,
      "mem_error_rate": 0.02,
      "triangle_centers": np.linspace(-1.0, 1.0, self.num_triangles),
      "weight_lr": weight_lr,
      "num_epochs": 10, #10 takes about 13 hours on rw1 with batch size 50
      "decay_rate": decay_rate,
      "staircase": staircase}]

  def set_data_params(self, data_type):
    self.data_type = data_type
    if data_type.lower() == "nat_images":
      self.model_name += "_nat_images"
    else:
      assert False, ("Data type "+data_type+" is not supported.")

  def set_test_params(self, data_type):
    self.model_name = "test_"+self.model_name
    self.set_data_params(data_type)
