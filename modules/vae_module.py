import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from DeepSparseCoding.ops.init_ops import L2NormalizedTruncatedNormalInitializer
import DeepSparseCoding.utils.entropy_functions as ef
from DeepSparseCoding.utils.trainable_variable_dict import TrainableVariableDict
from DeepSparseCoding.modules.ae_module import AeModule

class VaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, enc_channels, dec_channels, patch_size,
    conv_strides, mean_act_funcs, mean_layer_types, mean_channels, mean_patch_size,
    mean_conv_strides, mean_dropout, var_act_funcs, var_layer_types, var_channels,
    var_patch_size, var_conv_strides, var_dropout, w_decay_mult, w_norm_mult, kld_mult,
    act_funcs, dropout, tie_dec_weights, noise_level, prior_params, w_init_type,
    variable_scope="vae"):
    """
    Variational Autoencoder module
    Inputs:
      data_tensor
      layer_types [list of str] list of types for each layer, either "fc" or "conv"
      enc_channels [list of ints] the number of output channels per encoder layer
        Last entry is the number of latent units
      dec_channels [list of ints] the number of output channels per decoder layer
        Last entry must be the number of input pixels for FC layers and channels for CONV layers
      patch_size: number of (y, x) inputs for convolutional patches
      conv_strides [list] list of strides for convolution [batch, y, x, channels]
      mean_act_funcs - same format as act_funcs but for vae mean
      mean_layer_types - same format as layer_types but for vae mean
      mean_channels - same format as channels but for vae mean
      mean_patch_size - same format as patch_size but for vae mean
      var_act_funcs - same format as act_funcs but for vae var
      var_layer_types - same format as layer_types but for vae var
      var_channels - same format as channels but for vae var
      var_patch_size - same format as patch_size but for vae var
      w_decay_mult [float] tradeoff multiplier for weight decay loss
      w_norm_mult [float] tradeoff multiplier for weight norm loss (asks weight norm to == 1)
      kld_mult [float] tradeoff multiplier for latent variational kld loss
      act_funcs [list of functions] activation functions
      dropout [list of floats] specifies the keep probability or None
      noise_level [float] stddev of noise to be added to the input (for denoising VAE)
      prior_params [dict] containing keys for posterior & prior parameters
        posterior_prior [str] either "gauss_gauss" or "gauss_laplacian"
          posterior and prior used for latent KLD loss term
        gauss_mean [float]
        gauss_var [float]
        cauchy_location [float]
        cauchy_scale [float]
        laplace_scale [float]
      w_init_type: if True, l2 normalize w_init,
        reducing over [0] axis on enc and [-1] axis on dec
      variable_scope [str] specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.noise_level = noise_level
    with tf.compat.v1.variable_scope(variable_scope) as scope:
      if self.noise_level > 0.0:
          self.corrupt_data = tf.add(tf.random.normal(shape=tf.shape(data_tensor),
            mean=tf.reduce_mean(data_tensor), stddev=noise_level, dtype=tf.float32),
            data_tensor, name="noisy_data")
      else:
        self.corrupt_data = tf.identity(data_tensor, name="clean_data")
    self.prior_params = prior_params
    self.posterior_prior = self.prior_params["posterior_prior"]
    self.kld_mult = kld_mult
    self.mean_act_funcs = mean_act_funcs
    self.mean_layer_types = mean_layer_types
    self.mean_channels = mean_channels
    self.mean_patch_size_y = [int(size[0]) for size in mean_patch_size]
    self.mean_patch_size_x = [int(size[1]) for size in mean_patch_size]
    self.mean_conv_strides = mean_conv_strides
    self.mean_dropout = mean_dropout
    self.mean_num_fc_layers = self.mean_layer_types.count("fc")
    self.mean_num_conv_layers = self.mean_layer_types.count("conv")
    self.mean_num_layers = self.mean_num_fc_layers + self.mean_num_conv_layers
    self.var_act_funcs = var_act_funcs
    self.var_layer_types = var_layer_types
    self.var_channels = var_channels
    self.var_patch_size_y = [int(size[0]) for size in var_patch_size]
    self.var_patch_size_x = [int(size[1]) for size in var_patch_size]
    self.var_conv_strides = var_conv_strides
    self.var_dropout = var_dropout
    self.var_num_fc_layers = self.var_layer_types.count("fc")
    self.var_num_conv_layers = self.var_layer_types.count("conv")
    self.var_num_layers = self.var_num_fc_layers + self.var_num_conv_layers
    super(VaeModule, self).__init__(data_tensor, layer_types, enc_channels, dec_channels,
      patch_size, conv_strides, w_decay_mult, w_norm_mult, act_funcs, dropout, tie_dec_weights,
      w_init_type, variable_scope)
    assert len(self.var_channels) == len(self.mean_channels), (
      "len(self.var_channels) must equal len(self.mean_channels)")
    assert all([layer_type in ["conv", "fc"] for layer_type in mean_layer_types]), \
      ("All mean_layer_types must be conv or fc")
    assert all([layer_type in ["conv", "fc"] for layer_type in var_layer_types]), \
      ("All var_layer_types must be conv or fc")
    assert len(self.mean_act_funcs) == self.mean_num_layers, \
      ("mean_act_funcs parameter must be a list of size " + str(self.mean_num_layers))
    assert len(self.var_act_funcs) == self.var_num_layers, \
      ("var_act_funcs parameter must be a list of size " + str(self.var_num_layers))
    assert len(self.mean_channels) == self.mean_num_layers, \
      ("mean_act_funcs parameter must be a list of size " + str(self.mean_num_layers))
    assert len(self.var_channels) == self.var_num_layers, \
      ("var_act_funcs parameter must be a list of size " + str(self.var_num_layers))
    if self.enc_layer_types[-1] == "fc":
      assert np.all("fc" in self.mean_layer_types), (
        "vae_mean_layer_types must all be 'conv' if enc_layer_types[-1] == 'fc'")
      assert np.all("fc" in self.var_layer_types), (
        "vae_var_layer_types must all be 'conv' if enc_layer_types[-1] == 'fc'")

  def compute_latent_loss(self, act, dist_param1, dist_param2):
    with tf.compat.v1.variable_scope("latent"):
      reduc_dim = list(range(1, len(dist_param1.shape))) # Want to avg over batch, sum over the rest
      if self.posterior_prior.lower() == "gauss_gauss":
        mean = dist_param1
        logvar = dist_param2
        latent_loss = self.kld_mult * tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + logvar
          - tf.exp(logvar) - tf.square(mean), reduc_dim))
      elif self.posterior_prior.lower() == "gauss_laplacian":
        assert False, ("Not implemented.")
      elif self.posterior_prior.lower() == "cauchy_cauchy":
          #F Chyzak, F Nielsen (2019) -
          #  A Closed-Form Formula for the Kullback-Leibler Divergence Between Cauchy Distributions
          post_loc = dist_param1
          post_scale = dist_param2
          prior_loc = self.prior_params["cauchy_location"]
          prior_scale = self.prior_params["cauchy_scale"]
          latent_loss = self.kld_mult * tf.reduce_mean(tf.reduce_sum(tf.math.log(
            tf.divide(tf.square(post_scale + prior_scale) + tf.square(post_loc - prior_loc),
            4.0 * post_scale * prior_loc)), reduc_dim))
      else:
        assert False, (
          "posterior_prior parameter must be 'cauchy_cauchy', 'gauss_gauss', or 'gauss_laplacian', not %s"%(
          self.posterior_prior))
    return latent_loss

  def compute_total_loss(self):
    with tf.compat.v1.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "weight_norm_loss":self.compute_weight_norm_loss(),
        "latent_loss":self.compute_latent_loss(self.act, self.latent_mean, self.latent_logvar)}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

  def reparameterize(self, dist_param1, dist_param2, noise):
    if "gauss_" in self.posterior_prior: # posterior is gaussian
      mean = dist_param1
      logvar = dist_param2
      act = mean + tf.exp(0.5 * logvar) * noise
    elif "cauchy_" in self.posterior_prior: # posterior is gaussian
      loc = dist_param1
      scale = dist_param2
      act = loc +  scale * noise
    return act

  def gen_noise(self, noise_type, shape):
    if noise_type == "standard_normal":
      return tfp.distributions.Normal(loc=0., scale=1.).sample(sample_shape=shape)
    elif noise_type == "standard_cauchy":
      return tfp.distributions.Cauchy(loc=0., scale=1.).sample(sample_shape=shape)
    assert False, ("Noise type "+noise_type+" not supported")

  def build_vae_encoder(self, input_tensor, activation_functions, num_conv_layers,
    num_fc_layers, channels, patch_size_y, patch_size_x, dropout, name_suffix):
    # TODO: make regular ae module build_encoder act like this one? without member variables. could also redo decoder this way. It seems more general
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []
    prev_input_features = input_tensor.get_shape().as_list()[-1]
    # Make conv layers first
    for layer_id in range(num_conv_layers):
      w_shape = [patch_size_y[layer_id], patch_size_x[layer_id],
        int(prev_input_features), int(channels[layer_id])]
      u_out, w, b = self.layer_maker(layer_id, enc_u_list[layer_id],
        activation_functions[layer_id], w_shape, keep_prob=dropout[layer_id],
        conv=True, decode=False, tie_dec_weights=False, name_suffix=name_suffix)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(enc_channels[layer_id])
    # Make fc layers second
    for fc_layer_id in range(num_fc_layers):
      layer_id = fc_layer_id + num_conv_layers
      if fc_layer_id == 0: # Input needs to be reshaped to [batch, num_units] for FC layers
        in_tensor = self.flatten_feature_map(enc_u_list[-1])
        prev_input_features = in_tensor.get_shape().as_list()[1]
      else:
        in_tensor = enc_u_list[layer_id]
      w_shape = [int(prev_input_features), int(channels[layer_id])]
      u_out, w, b = self.layer_maker(layer_id, in_tensor, activation_functions[layer_id],
        w_shape, keep_prob=dropout[layer_id], conv=False, decode=False, tie_dec_weights=False,
        name_suffix=name_suffix)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(channels[layer_id])
    return enc_u_list, enc_w_list, enc_b_list

  def build_graph(self):
    with tf.compat.v1.variable_scope(self.variable_scope) as scope:
      with tf.compat.v1.variable_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01)
        self.w_normal_init = tf.contrib.layers.variance_scaling_initializer()
        self.w_xavier_init = tf.contrib.layers.xavier_initializer(uniform=False)
        self.w_normed_enc_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=0, epsilon=1e-12, dtype=tf.float32) #TODO: Fix axis to be general to conv layers
        self.w_normed_dec_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-5)

      self.u_list = [self.corrupt_data]
      self.w_list = []
      self.b_list = []
      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_enc_layers])
      self.u_list += enc_u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      mean_u_list, mean_w_list, mean_b_list = self.build_vae_encoder(self.u_list[-1],
        self.mean_act_funcs, self.mean_num_conv_layers, self.mean_num_fc_layers,
        self.mean_channels, self.mean_patch_size_y, self.mean_patch_size_x, self.mean_dropout,
        name_suffix="_mean")
      self.u_list += mean_u_list
      for mean_w, mean_b in zip(mean_w_list, mean_b_list):
        self.trainable_variables[mean_w.name] = mean_w
        self.trainable_variables[mean_b.name] = mean_b
      self.latent_mean = mean_u_list[-1]
      self.a = self.latent_mean #alias for AE model
      var_u_list, var_w_list, var_b_list = self.build_vae_encoder(self.u_list[-1],
        self.var_act_funcs, self.var_num_conv_layers, self.var_num_fc_layers,
        self.var_channels, self.var_patch_size_y, self.var_patch_size_x, self.var_dropout,
        name_suffix="_var")
      self.u_list += var_u_list
      for var_w, var_b in zip(var_w_list, var_b_list):
        self.trainable_variables[var_w.name] = var_w
        self.trainable_variables[var_b.name] = var_b
      self.latent_logvar = var_u_list[-1] # log(sigma**2)

      if "gauss_" in self.posterior_prior:
        noise_type = "standard_normal"
      elif "cauchy_" in self.posterior_prior:
        noise_type = "standard_cauchy"
      noise = self.gen_noise(noise_type, shape=tf.shape(self.latent_logvar))
      self.act = tf.identity(self.reparameterize(self.latent_mean,
        self.latent_logvar, noise), name="activity")
      self.u_list.append(self.act)

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_enc_layers:])
      self.u_list += dec_u_list
      self.w_list += dec_w_list
      self.b_list += dec_b_list

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.compat.v1.variable_scope("norm_weights") as scope:
        w_enc_norm_dim = list(range(len(self.w_list[0].get_shape().as_list())-1))
        self.norm_enc_w = self.w_list[0].assign(tf.nn.l2_normalize(self.w_list[0],
          axis=w_enc_norm_dim, epsilon=1e-8, name="row_l2_norm"))
        self.norm_dec_w = self.w_list[-1].assign(tf.nn.l2_normalize(self.w_list[-1],
          axis=-1, epsilon=1e-8, name="col_l2_norm"))
        self.norm_w = tf.group(self.norm_enc_w, self.norm_dec_w, name="l2_norm_weights")

      with tf.compat.v1.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      self.compute_total_loss()
