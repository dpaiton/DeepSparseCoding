import numpy as np
import tensorflow as tf
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from ops.init_ops import GDNGammaInitializer
from modules.activations import activation_picker

class DaeModule(AeModule):
  def __init__(self, data_tensor, output_channels, ent_mult, decay_mult, bounds_slope, latent_min,
    latent_max, num_triangles, mle_step_size, num_mle_steps, num_quant_bins, noise_var_mult, gdn_w_init_const, gdn_b_init_const,
    gdn_w_thresh_min, gdn_b_thresh_min, gdn_eps, act_funcs, dropout, tie_decoder_weights, conv=False, 
    conv_strides=None, patch_y=None, patch_x=None, name_scope="dae"):
    """
    Divisive Autoencoder module
    Inputs:
      data_tensor
      output_channels: A list of channels to make, also defines number of layers
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
      act_funcs: activation functions
      dropout: specifies the keep probability or None
      conv: if True, do convolution
      conv_strides: list of strides for convolution [batch, y, x, channels]
      patch_y: number of y inputs for convolutional patches
      patch_x: number of x inputs for convolutional patches
      name_scope: specifies the name_scope for the module
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
    super(DaeModule, self).__init__(data_tensor, output_channels, decay_mult, act_funcs,
      dropout, tie_decoder_weights, conv, conv_strides, patch_y, patch_x, name_scope)

  def compute_entropies(self, a_in):
    a_probs = ef.prob_est(a_in, self.mle_thetas, self.triangle_centers)
    a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
    return a_entropies

  def compute_entropy_loss(self, a_in):
    with tf.name_scope("latent"):
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

  def layer_maker(self, layer_id, input_tensor, activation_function, w_shape, decode):
    """
    Make layer that does act(u*w+b)
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

      w_name = "w_"+str(w_read_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      trainable_variables.append(w)

      b_name = "b_"+str(layer_id)
      if self.conv and decode:
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]
      b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32,
        initializer=self.b_init, trainable=True)
      trainable_variables.append(b)

      pre_act = self.compute_pre_activation(layer_id, input_tensor, w, b, decode)
      if activation_function == activation_picker("gdn"):
        w_gdn = tf.get_variable(name="w_gdn_"+str(layer_id), shape=[w_shape[-1], w_shape[-1]],
          dtype=tf.float32, initializer=self.w_gdn_init, trainable=True)
        trainable_variables.append(w_gdn)
        b_gdn = tf.get_variable(name="b_gdn_"+str(layer_id), shape=b_shape,
          dtype=tf.float32, initializer=self.b_gdn_init, trainable=True)
        trainable_variables.append(b_gdn)
        gdn_inverse = True if layer_id >= self.num_encoder_layers else False
        output_tensor, gdn_mult = activation_function(pre_act, w_gdn, b_gdn, self.gdn_w_thresh_min,
          self.gdn_b_thresh_min, self.gdn_eps, gdn_inverse, conv=self.conv)
      else:
        output_tensor = activation_function(pre_act)
      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, trainable_variables

  def build_encoder(self, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    u_list = [input_tensor]
    w_list = []
    b_list = []
    w_gdn_list = []
    b_gdn_list = []
    for layer_id in range(len(w_shapes)):
      u_out, trainable_variables = self.layer_maker(layer_id, u_list[layer_id],
        activation_functions[layer_id], w_shapes[layer_id], decode=False)
      if activation_functions[layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        w_gdn_list.append(w_gdn)
        b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      u_list.append(u_out)
      w_list.append(w)
      b_list.append(b)
    return u_list, w_list, b_list, w_gdn_list, b_gdn_list

  def build_decoder(self, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    u_list = [input_tensor]
    w_list = []
    b_list = []
    w_gdn_list = []
    b_gdn_list = []
    for dec_layer_id in range(len(w_shapes)):
      layer_id = self.num_encoder_layers + dec_layer_id
      u_out, trainable_variables = self.layer_maker(layer_id, u_list[dec_layer_id],
        activation_functions[dec_layer_id], w_shapes[dec_layer_id], decode=True)
      if activation_functions[dec_layer_id] == activation_picker("gdn"):
        w, b, w_gdn, b_gdn = trainable_variables
        w_gdn_list.append(w_gdn)
        b_gdn_list.append(b_gdn)
      else:
        w, b = trainable_variables
      u_list.append(u_out)
      w_list.append(w)
      b_list.append(b)
    return u_list, w_list, b_list, w_gdn_list, b_gdn_list

  def build_graph(self):
    with tf.name_scope(self.name_scope) as scope:
      with tf.name_scope("weight_inits") as scope:
        self.w_init = tf.initializers.random_normal(mean=0.0, stddev = 1e-2, dtype=tf.float32)
        self.b_init = tf.initializers.zeros(dtype=tf.float32)

      with tf.name_scope("gdn_weight_inits") as scope:
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
        self.build_encoder(self.u_list[0], self.act_funcs[:self.num_encoder_layers],
        self.w_shapes[:self.num_encoder_layers])
      self.u_list += enc_u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list
      self.w_gdn_list += enc_w_gdn_list
      self.b_gdn_list += enc_b_gdn_list

      with tf.name_scope("inference") as scope:
        self.a = tf.identity(self.u_list[-1], name="activity")

      with tf.variable_scope("probability_estimate") as scope:
        self.mle_thetas, self.theta_init = ef.construct_thetas(self.output_channels[-1],
          self.num_triangles)

        ll = ef.log_likelihood(tf.nn.sigmoid(tf.reshape(self.a, [tf.shape(self.a)[0], -1])),
          self.mle_thetas, self.triangle_centers)
        self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
          for _ in range(self.num_mle_steps)]

      noise_var = self.noise_var_mult*(self.latent_max-self.latent_min)/(2*self.num_quant_bins)
      noise = tf.random_uniform(shape=tf.stack(tf.shape(self.u_list[-1])),minval=-noise_var,
        maxval=noise_var)
      a_noise = tf.add(noise,self.u_list[-1])

      dec_u_list, dec_w_list, dec_b_list, dec_w_gdn_list, dec_b_gdn_list  = \
        self.build_decoder(a_noise, self.act_funcs[self.num_encoder_layers:],
        self.w_shapes[self.num_encoder_layers:])
      self.u_list += dec_u_list
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

      with tf.name_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      with tf.name_scope("loss") as scope:
        self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
          "weight_decay_loss":self.compute_weight_decay_loss(),
          "entropy_loss":self.compute_entropy_loss(self.a),
          "ramp_loss":self.compute_ramp_loss(self.a)}
        self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
