import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from modules.dae_module import DaeModule
from ops.init_ops import GDNGammaInitializer
from modules.activations import activation_picker
from utils import get_data
from utils import mem_utils

class DaeMemModule(DaeModule):
  def __init__(self, data_tensor, layer_types, output_channels, patch_size, conv_strides, ent_mult,
    decay_mult, bounds_slope, latent_min, latent_max, num_triangles, mle_step_size, num_mle_steps,
    gdn_w_init_const, gdn_b_init_const, gdn_w_thresh_min, gdn_b_thresh_min, gdn_eps,
    memristor_data_loc, memristor_type, memristor_std_eps, synthetic_noise, mem_error_rate,
    act_funcs, dropout, tie_decoder_weights, variable_scope="dae_mem"):
    """
    Divisive Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      ent_mult: tradeoff multiplier for latent entropy loss
      decay_mult: tradeoff multiplier for weight decay loss
      bounds_slope: slope for out of bounds loss (two relus back to back)
      latent_min: min value you want for latent variable (max value for left relu)
      latent_max: max value you want for latent variable (max value for right relu)
      num_triangles: number of triangle kernals to use for the entropy estimator
      mle_step_size: size of maximimum likelihood estimator steps
      num_mle_steps: number of max likelihood estimation steps for the entropy estimator
      num_quant_bins: number of bins you want for quantization
        e.g. if min is -50 and max is 50 and num_quant_bins is 100, will qauntize on integers
        formula: quant noise drawn from
        U(-(latent_max-latent_min)/(2*num_quant_bins), (latent_max-latent_min)/(2*num_quant_bins))
      noise_var_mult: multiplier to scale noise bounds that is added to the latent code
      gdn_w_init_const: diagonal of gdn gamma initializer
      gdn_b_init_const: diagonal of gdn beta initializer
      gdn_w_thresh_min: minimum allowable value for gdn_w
      gdn_b_thresh_min: minimum allowable value for gdn_b
      gdn_eps: off diagonal of gdn gamma initializer
      memristor_type: the type of memristor for memristorize
      synthetic_noise: noise to create synthetic channels (e.g. upper/lower bounds for RRAM data with write verify)
      act_funcs: activation functions
      dropout: specifies the keep probability or None
      conv: if True, do convolution
      conv_strides: list of strides for convolution [batch, y, x, channels]
      patch_y: number of y inputs for convolutional patches
      patch_x: number of x inputs for convolutional patches
      variable_scope: specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    #need to put more mem variables here#
    self.memristor_data_loc = memristor_data_loc
    self.memristor_type = memristor_type
    self.memristor_std_eps = memristor_std_eps
    self.synthetic_noise = synthetic_noise
    self.mem_error_rate = mem_error_rate
    noise_var_mult = 0 # for the mem module we only want memristor noise, not mem + quantization noise
    num_quant_bins = 10 # just setting this to something non-zero so there's no chance of breaking
    super(DaeMemModule, self).__init__(data_tensor, layer_types, output_channels, patch_size,
      conv_strides, ent_mult, decay_mult, bounds_slope, latent_min, latent_max, num_triangles,
      mle_step_size, num_mle_steps, num_quant_bins, noise_var_mult, gdn_w_init_const,
      gdn_b_init_const, gdn_w_thresh_min, gdn_b_thresh_min, gdn_eps, act_funcs, dropout,
      tie_decoder_weights, variable_scope)

  def memristorize(self, u_in, memristor_std_eps, memristor_type=None, synthetic_noise=None):
    if memristor_type is None:
      return u_in
    elif memristor_type == "gauss":
      get_channel_data = get_data.get_gauss_data
    elif memristor_type == "rram":
      get_channel_data = get_data.get_rram_data
    elif memristor_type == "pcm":
      get_channel_data = get_data.get_pcm_data
    else:
      assert False, ("memristor_type must be None, 'rram', 'gauss', or 'pcm'")
    u_in_shape = u_in.get_shape().as_list()
    if len(u_in_shape) == 4:
      n_mem = u_in_shape[1]*u_in_shape[2]*u_in_shape[3]
    elif len(u_in_shape) == 2:
      n_mem = u_in_shape[1]
    else:
      assert False, ("What's up with that shape mane??") 
    (vs_data, mus_data, sigs_data,
      orig_VMIN, orig_VMAX, orig_RMIN,
      orig_RMAX) = get_channel_data(self.memristor_data_loc, n_mem, num_ext=5,
      norm_min=self.latent_min, norm_max=self.latent_max, synthetic_noise=self.synthetic_noise)
    v_clip = tf.clip_by_value(u_in, clip_value_min=self.latent_min, clip_value_max=self.latent_max)
    r = mem_utils.memristor_output(v_clip, memristor_std_eps, vs_data, mus_data, sigs_data,
      interp_width=np.array(vs_data[1, 0] - vs_data[0, 0]).astype('float32'), error_rate = self.mem_error_rate)
    u_out = tf.reshape(r, shape=tf.shape(u_in), name="mem_r")
    return u_out



  def build_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev = 1e-2, dtype=tf.float32)
        self.b_init = tf.initializers.zeros(dtype=tf.float32)

      with tf.variable_scope("gdn_weight_inits") as scope:
        self.w_gdn_init = GDNGammaInitializer(diagonal_gain=self.gdn_w_init_const,
          off_diagonal_gain=self.gdn_eps, dtype=tf.float32)
        self.w_igdn_init = self.w_gdn_init
        b_init_const = np.sqrt(self.gdn_b_init_const + self.gdn_eps**2)
        self.b_gdn_init = tf.initializers.constant(b_init_const, dtype=tf.float32)
        self.b_igdn_init = self.b_gdn_init

      self.u_list = [self.data_tensor]
      self.w_list = []
      self.b_list = []
      self.w_gdn_list = []
      self.b_gdn_list = []
      enc_u_list, enc_w_list, enc_b_list, enc_w_gdn_list, enc_b_gdn_list = \
        self.build_encoder(self.u_list[0], self.act_funcs[:self.num_encoder_layers])
      self.enc_u_list = enc_u_list
      self.enc_w_list = enc_w_list
      self.u_list += enc_u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list
      self.w_gdn_list += enc_w_gdn_list
      self.b_gdn_list += enc_b_gdn_list

      u_shape = self.u_list[-1].get_shape().as_list()
      if self.layer_types[-1] == "conv":
        self.num_latent = np.prod(u_shape[1:])
      else:
        self.num_latent = self.output_channels[-1]

      with tf.variable_scope("inference") as scope:
        self.a = tf.identity(enc_u_list[-1], name="activity")

      with tf.variable_scope("probability_estimate") as scope:
        self.mle_thetas, self.theta_init = ef.construct_thetas(self.num_latent, self.num_triangles)

        ll = ef.log_likelihood(tf.nn.sigmoid(tf.reshape(self.a, [tf.shape(self.a)[0], -1])),
          self.mle_thetas, self.triangle_centers)
        self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
          for _ in range(self.num_mle_steps)]

      a_noise = self.memristorize(self.u_list[-1], self.memristor_std_eps, self.memristor_type, self.synthetic_noise)

      dec_u_list, dec_w_list, dec_b_list, dec_w_gdn_list, dec_b_gdn_list  = \
        self.build_decoder(a_noise, self.act_funcs[self.num_encoder_layers:])
      self.u_list += dec_u_list
      if not self.tie_decoder_weights:
        self.w_list += dec_w_list
      self.b_list += dec_b_list
      self.w_gdn_list += dec_w_gdn_list
      self.b_gdn_list += dec_b_gdn_list

      for w_gdn, b_gdn in zip(self.w_gdn_list, self.b_gdn_list):
        self.trainable_variables[w_gdn.name] = w_gdn
        self.trainable_variables[b_gdn.name] = b_gdn
      for w, b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      self.compute_total_loss()
