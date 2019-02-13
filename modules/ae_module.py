import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class AeModule(object):
  def __init__(self, data_tensor, output_channels, decay_mult, act_funcs, dropout,
    tie_decoder_weights, conv=False, conv_strides=None, patch_y=None, patch_x=None, name_scope="AE"):
    """
    Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      decay_mult: weight decay multiplier
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
    self.conv = conv
    self.conv_strides = conv_strides
    self.patch_y = patch_y
    self.patch_x = patch_x
    self.name_scope = name_scope

    data_ndim = len(data_tensor.get_shape().as_list())
    assert data_ndim == 2 or data_ndim == 4, (
      "Module requires datal_tensor to have shape [batch, num_pixels] or [batch, y, x, c]")

    self.data_tensor = data_tensor
    if data_ndim == 2:
      self.batch_size, self.num_pixels = self.data_tensor.get_shape()
    else:
      self.batch_size, self.num_pixels_y, self.num_pixels_x, self.num_channels = self.data_tensor.get_shape()
      self.num_pixels = self.num_pixels_y * self.num_pixels_x * self.num_channels

    self.output_channels = output_channels
    self.act_funcs = act_funcs
    self.dropout = dropout

    self.decay_mult = decay_mult

    self.trainable_variables = TrainableVariableDict()

    self.num_encoder_layers = len(self.output_channels)
    self.num_decoder_layers = self.num_encoder_layers
    self.num_layers = self.num_encoder_layers + self.num_decoder_layers
    self.tie_decoder_weights = tie_decoder_weights

    assert len(self.act_funcs) == self.num_layers, \
      ("act_funcs parameter must have the same length as output_channels")

    if self.conv:# Convoluational AE
      in_channels = [data_tensor.shape[-1]] + self.output_channels[:-1]
      self.w_shapes = [shape
        for shape in zip(self.patch_y, self.patch_x, in_channels, self.output_channels)]
      self.w_shapes += self.w_shapes[::-1] # decoder mirrors the encoder
    else:
      w_enc_shape = []
      w_dec_shape = []
      prev_input_features = self.num_pixels
      for l in range(self.num_encoder_layers):
        w_enc_shape.append([int(prev_input_features), int(self.output_channels[l])])
        prev_input_features = self.output_channels[l]
      w_dec_shape = [shape[::-1] for shape in w_enc_shape[::-1]]
      self.w_shapes = w_enc_shape + w_dec_shape

    self.build_graph()

  def compute_weight_decay_loss(self):
    with tf.name_scope("unsupervised"):
      w_decay_list = [tf.reduce_sum(tf.square(w)) for w in self.w_list]
      decay_loss = tf.multiply(0.5*self.decay_mult, tf.add_n(w_decay_list))
    return decay_loss

  def compute_recon_loss(self, reconstruction):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(reconstruction.shape)))# We want to avg over batch
      recon_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(reconstruction, self.data_tensor)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_pre_activation(self, layer_id, input_tensor, w, b, decode):
    if self.conv:
      if decode:
        height_const = 0 if tf.shape(input_tensor)[1] % self.conv_strides[layer_id][1] == 0 else 1
        out_height = (tf.shape(input_tensor)[1] * self.conv_strides[layer_id][1]) - height_const
        width_const = 0 if tf.shape(input_tensor)[2] % self.conv_strides[layer_id][2] == 0 else 1
        out_width = (tf.shape(input_tensor)[2] * self.conv_strides[layer_id][2]) - width_const
        out_shape = tf.stack([tf.shape(input_tensor)[0], # Batch
          out_height, # Height
          out_width, # Width
          tf.constant(w.get_shape()[2], dtype=tf.int32)]) # Channels
        pre_act =  tf.add(tf.nn.conv2d_transpose(input_tensor, w, out_shape,
          strides=self.conv_strides[layer_id], padding="SAME"), b)
      else:
        pre_act = tf.add(tf.nn.conv2d(input_tensor, w, self.conv_strides[layer_id],
          padding="SAME"), b)
    else:
      pre_act = tf.add(tf.matmul(input_tensor, w), b)
    return pre_act

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
    with tf.variable_scope("layer"+str(layer_id), reuse=tf.AUTO_REUSE) as scope:
      if self.tie_decoder_weights:
        w_read_id = self.num_layers - (layer_id+1)
      else:
        w_read_id = layer_id

      w_name = "w_"+str(w_read_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)

      b_name = "b_"+str(layer_id)
      if self.conv and decode:
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]
      b = tf.get_variable(name=b_name, shape=b_shape,
        dtype=tf.float32, initializer=self.b_init, trainable=True)

      pre_act = self.compute_pre_activation(layer_id, input_tensor, w, b, decode)
      output_tensor = activation_function(pre_act)
      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, w, b

  def build_encoder(self, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []
    for layer_id in range(len(w_shapes)):
      u_out, w, b = self.layer_maker(layer_id, enc_u_list[layer_id], activation_functions[layer_id],
        w_shapes[layer_id], decode=False)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
    return enc_u_list, enc_w_list, enc_b_list

  def build_decoder(self, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []
    for dec_layer_id in range(len(w_shapes)):
      layer_id = self.num_encoder_layers + dec_layer_id
      u_out, w, b = self.layer_maker(layer_id, dec_u_list[dec_layer_id],
        activation_functions[dec_layer_id], w_shapes[dec_layer_id], decode=True)
      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)
    return dec_u_list, dec_w_list, dec_b_list

  def build_graph(self):
    with tf.name_scope(self.name_scope) as scope:
      with tf.name_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01, dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

      self.u_list = [self.data_tensor]
      self.w_list = []
      self.b_list = []
      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_encoder_layers], self.w_shapes[:self.num_encoder_layers])
      self.u_list += enc_u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      with tf.variable_scope("inference") as scope:
        self.a = tf.identity(enc_u_list[-1], name="activity")

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_encoder_layers:], self.w_shapes[self.num_encoder_layers:])
      self.u_list += dec_u_list
      if not self.tie_decoder_weights:
        self.w_list += dec_w_list
      self.b_list += dec_b_list

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.name_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      with tf.name_scope("loss") as scope:
        self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
          "weight_decay_loss":self.compute_weight_decay_loss()}
        self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
