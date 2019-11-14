import numpy as np
import tensorflow as tf
from ops.init_ops import L2NormalizedTruncatedNormalInitializer
from utils.trainable_variable_dict import TrainableVariableDict

class AeModule(object):
  def __init__(self, data_tensor, layer_types, output_channels, patch_size, conv_strides,
    decay_mult, norm_mult, act_funcs, dropout, tie_decoder_weights, norm_w_init, variable_scope="ae"):
    """
    Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      decay_mult: tradeoff multiplier for weight decay loss
      norm_mult: tradeoff multiplier for weight norm loss (asks weight norm to == 1)
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
    self.conv_strides = conv_strides
    self.variable_scope = variable_scope
    self.trainable_variables = TrainableVariableDict()

    self.data_tensor = data_tensor
    self.layer_types = layer_types
    self.output_channels = output_channels
    self.patch_size_y = [size[0] for size in patch_size]
    self.patch_size_x = [size[1] for size in patch_size]
    self.conv_strides = conv_strides
    self.dropout = dropout
    self.decay_mult = decay_mult
    self.norm_mult = norm_mult
    self.act_funcs = act_funcs
    self.tie_decoder_weights = tie_decoder_weights

    self.num_conv_layers = layer_types.count("conv")
    self.num_fc_layers = layer_types.count("fc")
    self.num_encoder_layers = self.num_fc_layers + self.num_conv_layers
    self.num_decoder_layers = self.num_encoder_layers
    self.num_layers = self.num_encoder_layers + self.num_decoder_layers

    data_ndim = len(data_tensor.get_shape().as_list())
    if data_ndim == 2:
      self.batch_size, self.num_pixels = self.data_tensor.get_shape()
    else:
      self.batch_size, self.num_pixels_y, self.num_pixels_x, self.num_channels = \
        self.data_tensor.get_shape()
      self.num_pixels = self.num_pixels_y * self.num_pixels_x * self.num_channels

    if layer_types[0] == "conv":
      assert data_ndim == 4, (
        "Module requires data_tensor to have shape" +
        " [batch, num_pixels_y, num_pixels_x, num_features] if first layer is conv")
    else:
      assert data_ndim == 2, (
        "Module requires data_tensor to have shape [batch, num_pixels]")
    if self.num_conv_layers > 0:
      assert np.all("conv" in self.layer_types[:self.num_conv_layers]), \
        ("Conv layers must come before fc layers")
    assert len(self.layer_types) == self.num_encoder_layers, \
      ("All layer_types must be conv or fc")
    assert len(self.output_channels) == self.num_encoder_layers, \
      ("output_channels must be a list of size " + str(self.num_encoder_layers))
    assert len(self.patch_size_y) == self.num_conv_layers, \
      ("patch_size_y must be a list of size " + str(self.num_conv_layers))
    assert len(self.patch_size_x) == self.num_conv_layers, \
      ("patch_size_x must be a list of size " + str(self.num_conv_layers))
    assert len(self.conv_strides) == self.num_conv_layers, \
      ("conv_strides must be a list of size " + str(self.num_conv_layers))
    assert len(self.act_funcs) == self.num_layers, \
      ("act_funcs parameter must be a list of size " + str(self.num_layers))

    self.conv_strides = self.conv_strides + [None]*(self.num_fc_layers*2) + self.conv_strides[::-1]
    self.build_graph()

  def compute_weight_norm_loss(self):
    #with tf.variable_scope("w_norm"):
    #  num_neurons = self.output_channels + self.output_channels[::-1]
    #  w_norm_list = [tf.square(num_neurons[w_idx] - tf.reduce_sum(tf.square(w)))
    #    for w_idx, w in enumerate(self.w_list)]
    with tf.variable_scope("w_norm"):
      num_neurons = self.output_channels
      w_norm_list = []
      for w in self.w_list:
        reduc_axis = np.arange(1, len(w.get_shape().as_list()))
        w_norm = tf.reduce_sum(tf.square(1 - tf.reduce_sum(tf.square(w), axis=reduc_axis)))
        w_norm_list.append(w_norm)
      norm_loss = tf.multiply(0.5*self.norm_mult, tf.add_n(w_norm_list))
    return norm_loss

  def compute_weight_decay_loss(self):
    with tf.variable_scope("unsupervised"):
      w_decay_list = [tf.reduce_sum(tf.square(w)) for w in self.w_list]
      decay_loss = tf.multiply(0.5*self.decay_mult, tf.add_n(w_decay_list))
    return decay_loss

  def compute_recon_loss(self, reconstruction):
    with tf.variable_scope("unsupervised"):
      reduc_dim = list(range(1, len(reconstruction.shape)))# We want to avg over batch
      recon_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(reconstruction, self.data_tensor)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_total_loss(self):
    with tf.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "weight_norm_loss":self.compute_weight_norm_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

  def compute_pre_activation(self, layer_id, input_tensor, w, b, conv, decode):
    if conv:
      strides = self.conv_strides[layer_id]
      if decode:
        height_const = tf.shape(input_tensor)[1] % strides[1]
        out_height = (tf.shape(input_tensor)[1] * strides[1]) - height_const
        width_const = tf.shape(input_tensor)[2] % strides[2]
        out_width = (tf.shape(input_tensor)[2] * strides[2]) - width_const
        out_shape = tf.stack([tf.shape(input_tensor)[0], # Batch
          out_height, # Height
          out_width, # Width
          tf.shape(w)[2]]) # Channels
        pre_act = tf.add(tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides,
          padding="SAME"), b)
      else:
        pre_act = tf.add(tf.nn.conv2d(input_tensor, w, strides, padding="SAME"), b)
    else:
      pre_act = tf.add(tf.matmul(input_tensor, w), b)
    return pre_act

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
    with tf.variable_scope("layer"+str(layer_id), reuse=tf.AUTO_REUSE) as scope:
      if self.tie_decoder_weights:
        w_read_id = self.num_layers - (layer_id+1)
      else:
        w_read_id = layer_id

      name_prefix = "conv_" if conv else "fc_"
      w_name = name_prefix+"w_"+str(w_read_id)
      # TODO: params to switch init type
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_normal_init, trainable=True)

      #w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
      #  initializer=self.w_xavier_init, trainable=True)

      #if decode:
      #  w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
      #    initializer=self.w_normed_dec_init, trainable=True)
      #else:
      #  w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
      #    initializer=self.w_normed_enc_init, trainable=True)

      b_name = name_prefix+"b_"+str(layer_id)
      if conv and decode:
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]
      b = tf.get_variable(name=b_name, shape=b_shape,
        dtype=tf.float32, initializer=self.b_init, trainable=True)

      pre_act = self.compute_pre_activation(layer_id, input_tensor, w, b, conv, decode)
      output_tensor = activation_function(pre_act)
      #output_tensor = tf.nn.dropout(output_tensor, rate=1-self.dropout[layer_id])
      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, w, b

  def build_encoder(self, input_tensor, activation_functions):
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []
    prev_input_features = input_tensor.get_shape().as_list()[-1]
    # Make conv layers first
    for layer_id in range(self.num_conv_layers):
      w_shape = [int(self.patch_size_y[layer_id]), int(self.patch_size_x[layer_id]),
        int(prev_input_features), int(self.output_channels[layer_id])]
      u_out, w, b = self.layer_maker(layer_id, enc_u_list[layer_id],
        activation_functions[layer_id], w_shape, conv=True, decode=False)
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
      u_out, w, b = self.layer_maker(layer_id, in_tensor, activation_functions[layer_id],
        w_shape, conv=False, decode=False)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(self.output_channels[layer_id])
    return enc_u_list, enc_w_list, enc_b_list

  #TODO this decoder is asserting (in module's __init__) that the decoder mirrors the encoder
  # We would like this to not be true
  # Also, we currently need enc_u_list and enc_w_list for inferring conv input shapes and w_shape,
  #  we'd like to compute this in __init__ as self.w_shapes
  def build_decoder(self, input_tensor, activation_functions):
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []
    # Build FC layers first
    for dec_layer_id in range(self.num_fc_layers):
      layer_id = self.num_encoder_layers + dec_layer_id
      #Corresponding enc layer
      enc_w_id = -(dec_layer_id+1)
      w_shape = self.enc_w_list[enc_w_id].get_shape().as_list()[::-1]
      u_out, w, b = self.layer_maker(layer_id, dec_u_list[dec_layer_id],
        activation_functions[dec_layer_id], w_shape, conv=False, decode=True)
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
      u_out, w, b = self.layer_maker(layer_id, in_tensor, activation_functions[dec_conv_layer_id],
        w_shape, conv=True, decode=True)
      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)
    return dec_u_list, dec_w_list, dec_b_list

  def build_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope("weight_inits") as scope:
        self.w_normal_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.001, dtype=tf.float32)
        self.w_xavier_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
        self.w_normed_enc_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=0, epsilon=1e-12, dtype=tf.float32) #TODO: Fix axis to be general to conv layers
        self.w_normed_dec_init = L2NormalizedTruncatedNormalInitializer(mean=0.0, stddev=0.001,
          axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

      self.u_list = [self.data_tensor]
      self.w_list = []
      self.b_list = []
      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_encoder_layers])
      self.enc_u_list = enc_u_list # TODO: Remove the dependency for these member variables for build_decoder
      self.enc_w_list = enc_w_list
      self.u_list += enc_u_list[1:] # build_encoder() will place self.u_list[0] as enc_u_list[0]
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      with tf.variable_scope("inference") as scope:
        self.a = tf.identity(enc_u_list[-1], name="activity")

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_encoder_layers:])

      self.u_list += dec_u_list[1:] # build_decoder() will place self.u_list[-1] as dec_u_list[0]
      if not self.tie_decoder_weights:
        self.w_list += dec_w_list
      self.b_list += dec_b_list

      with tf.variable_scope("norm_weights") as scope:
        w_enc_norm_dim = list(range(len(self.w_list[0].get_shape().as_list())-1))
        self.norm_enc_w = self.w_list[0].assign(tf.nn.l2_normalize(self.w_list[0],
          axis=w_enc_norm_dim, epsilon=1e-8, name="row_l2_norm"))
        self.norm_dec_w = self.w_list[-1].assign(tf.nn.l2_normalize(self.w_list[-1],
          axis=-1, epsilon=1e-8, name="col_l2_norm"))
        self.norm_w = tf.group(self.norm_enc_w, self.norm_dec_w, name="l2_norm_weights")

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      self.compute_total_loss()
