import os
import numpy as np

class BaseParams(object):
  """
  all models
    batch_size [int] number of images in a training batch
    center_data [bool] if set, subrtract mean from dataset
    contrast_normalize [bool] if set, divide image pixels  by gaussian blurred surround pixels
    cp_int [int] how often to checkpoint
    cp_load [bool] if set, load from checkpoint
    cp_load_name [str] checkpoint model name to load
    cp_load_step [int] checkpoint time step to load
    cp_load_var [list of str] which model variables to load from checkpoint
                  if None or empty list, the model load all weights
    cp_load_ver [str] checkpoint version to load
    cp_set_var  [list of str] which variables to assign values to
                  len(cp_set_var) should equal len(cp_load_var)
    data_dir [str] location of dataset folders
    device [str] which device to run on
    eps [float] small value to avoid division by zero
    extract_patches [bool] if set, extract patches from dataset images during preprocessing
    gen_plot_int [int] interval (in batches) between plot outputs during training
      plot outputs are specified in the model's generate_plots function
    log_int [int] how often to send updates to stdout
    log_to_file [bool] if set, log to file, else log to stderr
    lpf_cutoff [float] between 0 and 1, the desired low-pass cutoff frequency (multiplied by nyquist)
    lpf_data [bool] if set, low pass filter dataset images using a Fourier filter
    max_cp_to_keep [int] how many checkpoints to keep. See max_to_keep tf arg
    model_name [str] name for model (can be anything)
    model_type [str] type of model (must be among the list returned by models/model_picker.get_model_list())
    norm_data [bool] if set, divide data by the maximum
    norm_weights [bool] if set, l2 normalize weights after each update
    num_images [int] how many images to use from the van Hateren dataset
    num_patches [int] how many patches to divide image up into
    num_pixels [int] total number of pixels in the input image
    optimizer [str] which optimization algorithm to use
                can be "annealed_sgd" (default) or "adam"
    out_dir [str] base directory for all model outputs
    overlapping_patches [bool] if set, extract patches from overlapping locations in the input images
      if False, patches are extracted by evenly tiling the input images
    patch_edge_size [int] number of pixels on the edge of a square patch to be extracted
      if overlapping_patches is false then this should divide evenly into the image edge size
    patch_variance_threshold [float] patches with variance below this value will be thrown out of the dataset constructor
    rand_seed [int] seed to be given to np.random.RandomState
    rand_state [int] random state to be used for all random functions to allow for reproducible results
    randomize_patches [bool] if set, the patches are drawn randomly from the image
      must be True if overlapping_patches is True
    rescale_data [bool] if set, rescale input data to be between 0 and 1, per example
    version [str] model version for output
              default is an empty string ""
    save_plots [bool] if set, save plots to file
    schedule [list of dicts] specifying values for auto_placeholders
      train_model.py will loop over each schedule dictionary in the list, which allows you to modify training parameters after some number of batches
      each model has different auto_placeholders; all of them must exist in schedule_dict.keys() for each dictionary in the schedule list
    standardize_data [bool] if set, z-score data to have mean=0 and standard deviation=1 using numpy operators
    tf_standardize_data [bool] if set, z-score data to have mean=0 and standard deviation=1 using tensorflow operators
      only compatible when vectorize_data is False because it must be done on 3d image
    vectorize_data [bool] if set, reshape input data to be [batch, data_dimensionality] (i.e. a batch of vectors)
      must be True for fully connected networks and False for convolutional networks
    version [str] version string to be appended to all outputs for training and analysis; can be any string, or empty
    whiten_data [bool] if set, perform whitening on input data. Whitening details are specified in models/base_model.preprocess_dataset()
    whiten_method [str] specify method for whitening; can be "FT", "ZCA", or "PCA"
    whiten_batch_size [int] batch size for whitening operation; use if dataset is too large to whiten all of it at once

  lca
    dt [float] discrete global time constant for neuron dynamics
      lca update rule is multiplied by dt/tau
    num_steps [int] number of lca inference steps to take
    rectify_a [bool] if set, rectify the layer 1 neuron activity
    tau [float] LCA time constant
      lca update rule is multiplied by dt/tau
    thresh_type [str] specifying LCA threshold function; can be "hard" or "soft"

  lca_conv
    image_edge_size [int] dataset images will be cropped to have this many pixels on each edge.
      None gives full-size.
      TODO: Implement on more datasets; currently works with vanhateren & field
    patch_size [_y / _x] [int] the y / x convolutional kernel size
    stride [_y / _x] [int] specifying the convoutional stride in the y / x direction
    use_tf_norm

  lca_subspace ; ica_subspace
    num_groups [int] number of groups to divide up weights into
      must evenly divide into num_neurons

  lca_pca
    num_pooling_units [int] number of second layer units

  lca ; rica
    num_neurons [int] how many neurons to use in latent representation

  mlp
    adversarial_attack_method [str] attack method for class adverarial attacks.
      allowable values are karakin_untargeted, kurakin_targeted, carlini_targeted
    adversarial_clip [bool] if True, clip final adversarial image to bound specified
    adversarial_clip_range [tuple or list] (min, max) float values for clipping final adversarial image
    adversarial_max_change [float] maximum allowable perturbation size (None has no limit)
    adversarial_num_steps [int] how many PGD adversarial steps to use
    adversarial_step_size [float] step size for PGD adversarial attack
    batch_norm [list of floats] batch norm coefficient; must be the same length as output_channels
    carlini_recon_mult [float] Tradeoff between input perturbation and target (between 0 and 1)
    lca_conv [bool] if set, use convolutional lca instead of fully-connected.
      only for mlp_lca_* models
    lrn [list of strings] specifies local response normalization
      string must be "pre" or "post" for performing local response normalization before or after pooling, respsectively
      None to skip lrn for that layer
      list must be same len as output_channels
    max_pool [list of bool] if set, perform max pooling
      must be the same len as output_channels
    max_pool_ksize [list of tuple] each tuple indicates the size of the pooling window for each dimension of the input tensor
      None if corresponding max_pool is False
      must be the same len as output_channels
    max_pool_strides [list of tuple] each tuple indicates the stride of the sliding window for each dimension of the input tensor
      None if corresponding max_pool is False
      must be the same len as output_channels
    mlp_output_channels [list of int] number of outputs per layer
    num_labeled [int] how many of the training images should be assigned a label (for semi-supervised training)
    val_on_cp [bool] if set, compute and log validation performance when checkpointing

  mlp; ae
    [mlp_ / ae_] activation_functions [list of str]  strings correspond to activation functions for layers.
      len must equal the len of output_channels
      strings must be one of those listed in  modules/activations.activation_picker()
    [mlp_ / ae_] conv_strides [list of tuples] containing strides for conv layers.
      Following the tf documentation for tf.nn.conv2d,
      the tuple contains [filter_height, filter_width, in_channels, out_channels]
    [mlp_ / ae_] dropout [list of floats] specifies dropout keep probability or None per layer
      len must be equal to the len of output_channels
    [mlp_ / ae_] layer_types [list of str] weight connectivity type, either "conv" or "fc"
      len must be equal to the len of output_channels
    [mlp_ / ae_] patch_size [list of tuples] each element in the list denotes conv (patch_y, patch_x)
      len must be equal to the len of output_channels
    norm_w_init [bool] if set, l2 normalize w_init

  mlp; lambda
    eval_batch_size [int] batch size for evaluating model on training data
    num_classes [int] number of label classes in supervised dataset
    num_val [int] number of validation images

  mlp_ae
    train_on_recon [bool] if set, train on autoencoder reconstructions; otherwise, train on autoencoder latent encodings

  ae
    enc_channels [list of ints] the number of output channels per encoder layer
      Last entry is the number of latent units
    dec_channels [list of ints] the number of output channels per decoder layer
      Last entry must be the number of input pixels for FC layers and channels for CONV layers
    mirror_dec_architecture [bool] if set, mirror the decoder architecture to match the encoder
      this will ignore all parameter entries that have index greater than len(enc_channels)
      each relevant parameter is set to be the mirror of the encoder portion
      parameters set (in ae_model) are: ae_activation_functions, ae_layer_types, ae_conv_strides, ae_patch_size, ae_dec_channels, ae_dropout
    num_data_channels [int] number of channels in the input
      this is used to set the architecture for testing
    num_edge_pixels [int] number of pixels in an edge of the square input
      this is used to set the architecture for testing.
      if the input is vectorized, then this should equal the square root of the input length
    tie_dec_weights [bool] if set, the decoder weights will equal the transpose of the encoder weights

  vae
    noise_level [float] standard deviation of noise added to the input data for denoising vae
    recon_loss_type [str] specify the reconstruction loss to be "mse" or "crossentropy"
      crossentropy assumes binary inputs

  dae
    bounds_slope [float] slope for out of bounds loss (two relus back to b ack)
    gdn_b_init_const [float] initial value for the GDN biases
    gdn_b_thresh_min [float] minimum allowable value for GDN biases
    gdn_eps [float] epsiolon that will be added to the GDN denominator
    gdn_w_init_const [float] inital value for the GDN weights
    gdn_w_thresh_min [float] minimum allowable value for the GDN weights
    latent_max [float] maximum allowable value for latent variables where ramp loss = 0
    latent_min [float] minimum allowable value for latent variables wehre ramp loss = 0
    mle_step_size [float] size of the maximum likelihood estimator steps
    num_mle_steps [int] number of maximum likelihood estimation steps for the entropy estimator
    num_quant_bins [int] number of bins you want for quantization
    num_triangles [int] number of triangle kernels to use for the entropy estimator

  dae_mem
    mem_error_rate [float] keep probability (1 - rate for tf.nn.dropout()) for memristors failing
    memristor_data_loc [str] location of pkl file containing memristor I/O data
    memristor_type [str] type of memristor to use. Can be "gauss", "rram", "pcm" or None.
      if using "rram", the std_eps will parameter will set the width of the uniform noise instead of the Gaussian multiplier
    synthetic_noise [float] noise to create synthetic channels (e.g. upper/lower bounds for RRAM data with write verify)

  ica
    prior [str] prior for ICA - can be "laplacian" or "cauchy"

  lambda
    activation_function [tf operation] to be used by lambda_model as the activation function

  rica
    maxiter [int] the maximum number of iterations for the lbfgs optimizer

  lista
    num_layers [int] how many encoding layers to use to approximate sparse coding
  """

  def __init__(self):
    self.vectorize_data = True
    self.norm_data = False
    self.rescale_data = True
    self.center_data = False
    self.standardize_data = False
    self.tf_standardize_data = False
    self.contrast_normalize = False
    self.whiten_data = False
    self.lpf_data = False
    self.lpf_cutoff = 0.7
    self.extract_patches = False
    self.num_images = 150
    self.batch_size = 100
    self.optimizer = "annealed_sgd"
    self.norm_weights = False
    self.cp_int = 10000
    self.max_cp_to_keep = 1
    self.cp_load = False
    self.val_on_cp = False
    self.cp_load_name = "pretrain"
    self.cp_load_step = None
    self.cp_load_ver = "0.0"
    self.cp_load_var = None
    self.log_int = 100
    self.log_to_file = True
    self.gen_plot_int = 10000
    self.save_plots = True
    self.eps = 1e-12
    self.device = "/gpu:0"
    self.rand_seed = 123456789
    self.rand_state = np.random.RandomState(self.rand_seed)
    self.out_dir = os.path.expanduser("~")+"/Work/Projects/"
    self.data_dir = os.path.expanduser("~")+"/Work/Datasets/"

  #def set_data_params(self, data_type):
  #  pass

  #def set_test_params(self, data_type):
  #  pass

