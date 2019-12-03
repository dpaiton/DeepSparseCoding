import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from ops.init_ops import GDNGammaInitializer
from modules.activations import activation_picker

class DaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, enc_channels, dec_channels, patch_size, conv_strides, ent_mult,
    w_decay_mult, w_norm_mult, bounds_slope, latent_min, latent_max, num_triangles, mle_step_size,
    num_mle_steps, num_quant_bins, noise_var_mult, gdn_w_init_const, gdn_b_init_const,
    gdn_w_thresh_min, gdn_b_thresh_min, gdn_eps, act_funcs, dropout, tie_dec_weights,
    norm_w_init, variable_scope="dae"):
    """
    Divisive Autoencoder module
    Inputs:
      data_tensor
      enc_channels [list of ints] the number of output channels per encoder layer
        Last entry is the number of latent units
      dec_channels [list of ints] the number of output channels per decoder layer
        Last entry must be the number of input pixels for FC layers and channels for CONV layers
      ent_mult: tradeoff multiplier for latent entropy loss
      w_decay_mult: tradeoff multiplier for weight decay loss
      w_norm_mult: tradeoff multiplier for weight norm loss (asks weight norm to == 1)
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
      act_funcs: activation functions
      dropout: specifies the keep probability or None
      conv: if True, do convolution
      conv_strides: list of strides for convolution [batch, y, x, channels]
      patch_size: number of (y, x) inputs for convolutional patches
      norm_w_init: if True, l2 normalize w_init,
        reducing over [0] axis on enc and [-1] axis on dec
      variable_scope: specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.ent_mult = ent_mult
    self.bounds_slope = bounds_slope
    self.latent_min = latent_min
    self.latent_max = latent_max
    self.num_triangles = num_triangles
    self.triangle_centers = np.linspace(self.latent_min, self.latent_max, self.num_triangles).astype(np.float32)
    self.mle_step_size = mle_step_size
    self.num_mle_steps = num_mle_steps
    self.num_quant_bins = num_quant_bins
    self.noise_var_mult = noise_var_mult
    self.gdn_w_init_const = gdn_w_init_const
    self.gdn_b_init_const = gdn_b_init_const
    self.gdn_w_thresh_min = gdn_w_thresh_min
    self.gdn_b_thresh_min = gdn_b_thresh_min
    self.gdn_eps = gdn_eps
    super(DaeModule, self).__init__(data_tensor, layer_types, enc_channels, dec_channels,
      patch_size, conv_strides, w_decay_mult, w_norm_mult, act_funcs, dropout, tie_dec_weights,
      norm_w_init, variable_scope)

  def compute_entropies(self, a_in):
    a_probs = ef.prob_est(a_in, self.mle_thetas, self.triangle_centers)
    a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
    return a_entropies

  def compute_entropy_loss(self, a_in):
    with tf.compat.v1.variable_scope("latent"):
      a_flat = self.flatten_feature_map(a_in)
      a_flat_sig = activation_picker("sigmoid")(a_flat)
      a_entropies = self.compute_entropies(a_flat_sig)
      entropy_loss = tf.multiply(self.ent_mult, tf.reduce_sum(a_entropies), name="entropy_loss")
    return entropy_loss

  def compute_ramp_loss(self, a_in):
    reduc_dim = list(range(1,len(a_in.shape))) # Want to avg over batch
    ramp_loss = tf.reduce_mean(tf.reduce_sum(self.bounds_slope
      * (tf.nn.relu(a_in - self.latent_max)
      + tf.nn.relu(self.latent_min - a_in)), axis=reduc_dim))
    return ramp_loss

  def compute_total_loss(self):
    with tf.compat.v1.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "weight_norm_loss":self.compute_weight_norm_loss(),
        "entropy_loss":self.compute_entropy_loss(self.a),
        "ramp_loss":self.compute_ramp_loss(self.a)}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

  def layer_maker(self, layer_id, input_tensor, activation_function, w_shape,
    conv=False, decode=False):
    """
    Make layer that does act(u*w+b) where * is a dot product or convolution
    Example case for w_read_id logic:
      layer_id: [0 1 2 3 4] [5 6 7 8 9]
                              10-6  10-7  10-8 10-9  10-10
      weight_id: [0 1 2 3 4] [ 4     3     2     1     0 ]
      num_layers: 10
      weight_id = num_layers - (layer_id + 1)
    """
    trainable_variables = []
    with tf.compat.v1.variable_scope("layer"+str(layer_id), reuse=tf.compat.v1.AUTO_REUSE) as scope:
      if self.tie_dec_weights:
        w_read_id = self.num_layers - (layer_id + 1)
      else:
        w_read_id = layer_id
      name_prefix = "conv_" if conv else "fc_"
      w_name = name_prefix+"w_"+str(w_read_id)
      w = tf.compat.v1.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      trainable_variables.append(w)
      b_name = name_prefix + "b_" + str(layer_id)
      if conv and decode:
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]
      b = tf.compat.v1.get_variable(name=b_name, shape=b_shape,
        dtype=tf.float32, initializer=self.b_init, trainable=True)
      trainable_variables.append(b)
      pre_act = self.compute_pre_activation(layer_id, input_tensor, w, b, conv, decode)
      if activation_function == activation_picker("gdn"):
        if conv and decode:
          w_gdn_shape = [w_shape[-2], w_shape[-2]]
        else:
          w_gdn_shape = [w_shape[-1], w_shape[-1]]
        w_gdn = tf.compat.v1.get_variable(name="w_gdn_"+str(layer_id), shape=w_gdn_shape,
          dtype=tf.float32, initializer=self.w_gdn_init, trainable=True)
        trainable_variables.append(w_gdn)
        b_gdn = tf.compat.v1.get_variable(name="b_gdn_"+str(layer_id), shape=b_shape,
          dtype=tf.float32, initializer=self.b_gdn_init, trainable=True)
        trainable_variables.append(b_gdn)
        output_tensor, gdn_mult = activation_function(pre_act, w_gdn, b_gdn, self.gdn_w_thresh_min,
          self.gdn_b_thresh_min, self.gdn_eps, decode, conv)
      else:
        output_tensor = activation_function(pre_act)
      output_tensor = tf.nn.dropout(output_tensor, rate=1-self.dropout[layer_id])
      #output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, trainable_variables

  def build_encoder(self, input_tensor, activation_functions):
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []
    enc_w_gdn_list = []
    enc_b_gdn_list = []
    prev_input_features = input_tensor.get_shape().as_list()[-1]
    # Make conv layers first
    for layer_id in range(self.num_enc_conv_layers):
      w_shape = [self.patch_size_y[layer_id], self.patch_size_x[layer_id],
        int(prev_input_features), int(self.enc_channels[layer_id])]
      u_out, trainable_variables = self.layer_maker(layer_id, enc_u_list[layer_id],
        activation_functions[layer_id], w_shape, conv=True, decode=False)
      if activation_functions[layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        enc_w_gdn_list.append(w_gdn)
        enc_b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(self.enc_channels[layer_id])
    # Make fc layers second
    for enc_fc_layer_id in range(self.num_enc_fc_layers):
      layer_id = enc_fc_layer_id + self.num_enc_conv_layers
      if enc_fc_layer_id == 0: # Input needs to be reshaped to [batch, num_units] for FC layers
        in_tensor = self.flatten_feature_map(enc_u_list[-1])
        prev_input_features = in_tensor.get_shape().as_list()[1]
      else:
        in_tensor = enc_u_list[layer_id]
      w_shape = [int(prev_input_features), int(self.enc_channels[layer_id])]
      u_out, trainable_variables = self.layer_maker(layer_id, in_tensor,
        activation_functions[layer_id], w_shape, conv=False, decode=False)
      if activation_functions[layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        enc_w_gdn_list.append(w_gdn)
        enc_b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(self.enc_channels[layer_id])
    return enc_u_list, enc_w_list, enc_b_list, enc_w_gdn_list, enc_b_gdn_list

  def build_decoder(self, input_tensor, activation_functions):
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []
    dec_w_gdn_list = []
    dec_b_gdn_list = []
    # Build FC layers first
    for dec_layer_id in range(self.num_dec_fc_layers):
      layer_id = self.num_enc_layers + dec_layer_id
      input_shape = dec_u_list[dec_layer_id].get_shape()
      if input_shape.ndims == 4: # if final enc layer was conv then flatten
        in_tensor = self.flatten_feature_map(dec_u_list[dec_layer_id])
      else: # final enc layer was fc
        in_tensor = dec_u_list[dec_layer_id]
      if dec_layer_id == self.num_dec_fc_layers - 1 and self.num_dec_conv_layers > 0:
        # If there are decoder conv layers, then
        # the last decoder FC layer needs to output a vector of the correct length
        # correct_length = feature_map_y * feature_map_x * feature_map_f
        # where feature_map_f = self.dec_channels[dec_layer_id]
        conv_layer_shapes = self.get_dec_shapes(self.data_tensor.get_shape())
        out_channels = np.prod(conv_layer_shapes[0])
      else:
        out_channels = self.dec_channels[dec_layer_id]
      w_shape = [in_tensor.get_shape()[-1], out_channels]
      u_out, trainable_variables = self.layer_maker(layer_id, in_tensor,
        activation_functions[dec_layer_id], w_shape, conv=False, decode=True)
      if activation_functions[dec_layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        dec_w_gdn_list.append(w_gdn)
        dec_b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)
    # Build conv layers second
    for dec_conv_layer_id in range(self.num_dec_conv_layers):
      dec_layer_id = self.num_dec_fc_layers + dec_conv_layer_id
      layer_id = self.num_enc_layers + dec_layer_id
      input_shape = dec_u_list[dec_layer_id].get_shape()
      if input_shape.ndims == 4: # prev layer was conv
        (batch, y, x, f) = input_shape
        in_tensor = dec_u_list[dec_layer_id]
        w_shape = [
          self.patch_size_y[self.num_enc_conv_layers + dec_conv_layer_id],
          self.patch_size_x[self.num_enc_conv_layers + dec_conv_layer_id],
          self.dec_channels[dec_layer_id],
          f]
      else: # prev layer was fc
        conv_layer_shapes = self.get_dec_shapes(self.data_tensor.get_shape())
        new_shape = [-1] + conv_layer_shapes[dec_conv_layer_id]
        in_tensor = tf.reshape(dec_u_list[dec_layer_id], new_shape)
        w_shape = [
          self.patch_size_y[self.num_enc_conv_layers + dec_conv_layer_id],
          self.patch_size_x[self.num_enc_conv_layers + dec_conv_layer_id],
          self.dec_channels[dec_layer_id],
          new_shape[-1]]
      u_out, trainable_variables = self.layer_maker(layer_id, in_tensor, activation_functions[dec_conv_layer_id],
        w_shape, conv=True, decode=True)
      if activation_functions[dec_layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        dec_w_gdn_list.append(w_gdn)
        dec_b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)
    return dec_u_list, dec_w_list, dec_b_list, dec_w_gdn_list, dec_b_gdn_list

  def build_graph(self):
    with tf.compat.v1.variable_scope(self.variable_scope) as scope:
      with tf.compat.v1.variable_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev = 1e-2)
        self.b_init = tf.initializers.zeros()
      with tf.compat.v1.variable_scope("gdn_weight_inits") as scope:
        self.w_gdn_init = GDNGammaInitializer(diagonal_gain=self.gdn_w_init_const,
          off_diagonal_gain=self.gdn_eps, dtype=tf.float32)
        self.w_igdn_init = self.w_gdn_init
        b_init_const = np.sqrt(self.gdn_b_init_const + self.gdn_eps**2)
        self.b_gdn_init = tf.initializers.constant(b_init_const)
        self.b_igdn_init = self.b_gdn_init
      self.u_list = [self.data_tensor]
      self.w_list = []
      self.b_list = []
      self.w_gdn_list = []
      self.b_gdn_list = []
      enc_u_list, enc_w_list, enc_b_list, enc_w_gdn_list, enc_b_gdn_list = \
        self.build_encoder(self.u_list[0], self.act_funcs[:self.num_enc_layers])
      self.u_list += enc_u_list[1:] # build_encoder() will place self.u_list[0] as enc_u_list[0]
      self.w_list += enc_w_list
      self.b_list += enc_b_list
      self.w_gdn_list += enc_w_gdn_list
      self.b_gdn_list += enc_b_gdn_list
      if self.enc_layer_types[-1] == "conv":
        self.num_latent = int(np.prod(self.u_list[-1].get_shape()[1:]))
      else:
        self.num_latent = self.enc_channels[-1]
      with tf.compat.v1.variable_scope("inference") as scope:
        self.a = tf.identity(enc_u_list[-1], name="activity")
      with tf.compat.v1.variable_scope("probability_estimate") as scope:
        self.mle_thetas, self.theta_init = ef.construct_thetas(self.num_latent, self.num_triangles)
        ll = ef.log_likelihood(tf.nn.sigmoid(tf.reshape(self.a, [tf.shape(self.a)[0], -1])),
          self.mle_thetas, self.triangle_centers)
        self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
          for _ in range(self.num_mle_steps)]
      noise_var = self.noise_var_mult*(self.latent_max-self.latent_min)/(2*self.num_quant_bins)
      noise = tf.random.uniform(shape=tf.stack(tf.shape(self.u_list[-1])), minval=-noise_var,
        maxval=noise_var)
      a_noise = tf.add(noise, self.a)
      dec_u_list, dec_w_list, dec_b_list, dec_w_gdn_list, dec_b_gdn_list  = \
        self.build_decoder(a_noise, self.act_funcs[self.num_enc_layers:])
      self.u_list += dec_u_list[1:] # build_decoder() will place self.u_list[-1] as dec_u_list[0]
      if not self.tie_dec_weights:
        self.w_list += dec_w_list
      self.b_list += dec_b_list
      self.w_gdn_list += dec_w_gdn_list
      self.b_gdn_list += dec_b_gdn_list
      with tf.compat.v1.variable_scope("norm_weights") as scope:
        w_enc_norm_dim = list(range(len(self.w_list[0].get_shape().as_list())-1))
        self.norm_enc_w = self.w_list[0].assign(tf.nn.l2_normalize(self.w_list[0],
          axis=w_enc_norm_dim, epsilon=1e-8, name="row_l2_norm"))
        self.norm_dec_w = self.w_list[-1].assign(tf.nn.l2_normalize(self.w_list[-1],
          axis=-1, epsilon=1e-8, name="col_l2_norm"))
        self.norm_w = tf.group(self.norm_enc_w, self.norm_dec_w, name="l2_norm_weights")
      for w_gdn, b_gdn in zip(self.w_gdn_list, self.b_gdn_list):
        self.trainable_variables[w_gdn.name] = w_gdn
        self.trainable_variables[b_gdn.name] = b_gdn
      for w, b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b
      with tf.compat.v1.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")
      self.compute_total_loss()
