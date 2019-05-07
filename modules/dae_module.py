import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from ops.init_ops import GDNGammaInitializer
from modules.activations import activation_picker

class DaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, output_channels, patch_size, conv_strides, ent_mult,
    decay_mult, norm_mult, bounds_slope, latent_min, latent_max, num_triangles, mle_step_size,
    num_mle_steps, num_quant_bins, noise_var_mult, gdn_w_init_const, gdn_b_init_const,
    gdn_w_thresh_min, gdn_b_thresh_min, gdn_eps, act_funcs, dropout, tie_decoder_weights,
    norm_w_init, variable_scope="dae"):
    """
    Divisive Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      ent_mult: tradeoff multiplier for latent entropy loss
      decay_mult: tradeoff multiplier for weight decay loss
      norm_mult: tradeoff multiplier for weight norm loss (asks weight norm to == 1)
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
    #if layer_types[-1] == "conv":
    #  self.latent_conv = True
    #else:
    #  self.latent_conv = False
    super(DaeModule, self).__init__(data_tensor, layer_types, output_channels, patch_size,
      conv_strides, decay_mult, norm_mult, act_funcs, dropout, tie_decoder_weights, norm_w_init,
      variable_scope)

  def compute_entropies(self, a_in):
    a_probs = ef.prob_est(a_in, self.mle_thetas, self.triangle_centers)
    a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
    return a_entropies

  def compute_entropy_loss(self, a_in):
    with tf.variable_scope("latent"):
      #a_flat = tf.reshape(a_in, [tf.shape(a_in)[0], -1])
      #a_flat = tf.layers.Flatten()(a_in)#tf.reshape(a_in, [tf.shape(a_in)[0], -1])
      dim = tf.reduce_prod(tf.shape(a_in)[1:])
      a_flat = tf.reshape(a_in, [tf.shape(a_in)[0], dim])
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
    with tf.variable_scope("loss") as scope:
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
    with tf.variable_scope("layer"+str(layer_id), reuse=tf.AUTO_REUSE) as scope:
      if self.tie_decoder_weights:
        w_read_id = self.num_layers - (layer_id+1)
      else:
        w_read_id = layer_id

      name_prefix = "conv_" if conv else "fc_"
      w_name = name_prefix+"w_"+str(w_read_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      trainable_variables.append(w)

      b_name = name_prefix+"b_"+str(layer_id)
      if conv and decode:
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]
      b = tf.get_variable(name=b_name, shape=b_shape,
        dtype=tf.float32, initializer=self.b_init, trainable=True)
      trainable_variables.append(b)

      pre_act = self.compute_pre_activation(layer_id, input_tensor, w, b, conv, decode)
      if activation_function == activation_picker("gdn"):
        #if self.latent_conv and decode:
        if conv and decode:
          w_gdn_shape = [w_shape[-2], w_shape[-2]]
        else:
          w_gdn_shape = [w_shape[-1], w_shape[-1]]
        w_gdn = tf.get_variable(name="w_gdn_"+str(layer_id), shape=w_gdn_shape,
          dtype=tf.float32, initializer=self.w_gdn_init, trainable=True)
        trainable_variables.append(w_gdn)
        b_gdn = tf.get_variable(name="b_gdn_"+str(layer_id), shape=b_shape,
          dtype=tf.float32, initializer=self.b_gdn_init, trainable=True)
        trainable_variables.append(b_gdn)
        #gdn_inverse = True if layer_id >= self.num_encoder_layers else False
        #output_tensor, gdn_mult = activation_function(pre_act, w_gdn, b_gdn, self.gdn_w_thresh_min,
        #  self.gdn_b_thresh_min, self.gdn_eps, gdn_inverse, conv=self.latent_conv)
        output_tensor, gdn_mult = activation_function(pre_act, w_gdn, b_gdn, self.gdn_w_thresh_min,
          self.gdn_b_thresh_min, self.gdn_eps, decode, conv)
      else:
        output_tensor = activation_function(pre_act)
      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, trainable_variables

  def build_encoder(self, input_tensor, activation_functions):
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []
    enc_w_gdn_list = []
    enc_b_gdn_list = []
    prev_input_features = input_tensor.get_shape().as_list()[-1]
    # Make conv layers first
    for layer_id in range(self.num_conv_layers):
      w_shape = [int(self.patch_size_y[layer_id]), int(self.patch_size_x[layer_id]),
        int(prev_input_features), int(self.output_channels[layer_id])]
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
      prev_input_features = int(self.output_channels[layer_id])

    # Make fc layers second
    for fc_layer_id in range(self.num_fc_layers):
      layer_id = fc_layer_id + self.num_conv_layers

      if fc_layer_id == 0:
        # Input needs to be reshaped to [batch, num_units] for FC layers
        enc_shape = enc_u_list[-1].get_shape().as_list()
        if len(enc_shape) == 4:
          (batch, y, x, f) = enc_shape
          prev_input_features = y * x * f # Flatten input (input_tensor or last conv layer)
          in_tensor  = tf.reshape(enc_u_list[-1], [-1, prev_input_features])
        elif(len(enc_shape) == 2):
          in_tensor = enc_u_list[-1]
        else:
          assert False, ("Final conv encoder output or input_tensor has incorrect ndim")
      else:
        in_tensor = enc_u_list[layer_id]

      w_shape = [int(prev_input_features), int(self.output_channels[layer_id])]
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
      prev_input_features = int(self.output_channels[layer_id])
    return enc_u_list, enc_w_list, enc_b_list, enc_w_gdn_list, enc_b_gdn_list

  def build_decoder(self, input_tensor, activation_functions):
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []
    dec_w_gdn_list = []
    dec_b_gdn_list = []
    # Build FC layers first
    for dec_layer_id in range(self.num_fc_layers):
      layer_id = self.num_encoder_layers + dec_layer_id
      #Corresponding enc layer
      enc_w_id = -(dec_layer_id+1)
      w_shape = self.enc_w_list[enc_w_id].get_shape().as_list()[::-1]
      u_out, trainable_variables = self.layer_maker(layer_id, dec_u_list[dec_layer_id],
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
    for dec_conv_layer_id in range(self.num_conv_layers):
      dec_layer_id = self.num_fc_layers + dec_conv_layer_id
      layer_id = self.num_encoder_layers + dec_layer_id

      if dec_conv_layer_id == 0:
        #Reshape flat vector for next conv
        u_list_id = -(self.num_fc_layers + 1)
        enc_shape = self.enc_u_list[u_list_id].get_shape().as_list()
        if len(enc_shape) == 4:
          (batch, y, x, f) = self.enc_u_list[u_list_id].get_shape().as_list()
          in_tensor = tf.reshape(dec_u_list[-1], [-1, y, x, f])
        else:
          in_tensor = dec_u_list[-1]
      else:
        u_list_id = -(dec_layer_id + 1) # u_list_id is the id for the INPUT of this layer
        in_tensor = dec_u_list[dec_layer_id]

      enc_w_id = -(dec_layer_id + 1)
      w_shape = self.enc_w_list[enc_w_id].get_shape().as_list()
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

      if self.layer_types[-1] == "conv":
        self.num_latent = tf.reduce_prod(self.u_list[-1].get_shape()[1:])
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

      noise_var = self.noise_var_mult*(self.latent_max-self.latent_min)/(2*self.num_quant_bins)
      noise = tf.random_uniform(shape=tf.stack(tf.shape(self.u_list[-1])),minval=-noise_var,
        maxval=noise_var)
      a_noise = tf.add(noise,self.u_list[-1])

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
