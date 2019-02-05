import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class AeModule(object):
  def __init__(self, data_tensor, output_channels, decay_mult, act_funcs, dropout,
    tie_decoder_weights):
    """
    Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      decay_mult: weight decay multiplier
      act_funcs: activation functions
      dropout: specifies the keep probability or None
    Outputs:
      dictionary
    """
    data_ndim = len(data_tensor.get_shape().as_list())
    assert data_ndim == 2, (
      "Module requires datal_tensor to have shape [batch, num_pixels]")

    self.data_tensor = data_tensor
    self.batch_size, self.num_pixels = self.data_tensor.get_shape()

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

  def layer_maker(self, layer_id, input_tensor, activation_function, w_shape):
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
      b = tf.get_variable(name=b_name, shape=w_shape[1],
        dtype=tf.float32, initializer=self.b_init, trainable=True)

      output_tensor = activation_function(tf.add(tf.matmul(input_tensor, w), b))
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
        w_shapes[layer_id])
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
    layer_id = self.num_encoder_layers
    for dec_layer_id in range(len(w_shapes)):
      u_out, w, b = self.layer_maker(layer_id, dec_u_list[dec_layer_id],
        activation_functions[dec_layer_id], w_shapes[dec_layer_id])
      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)
      layer_id += 1
    return dec_u_list, dec_w_list, dec_b_list

  def build_graph(self):
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
