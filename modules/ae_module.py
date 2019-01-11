import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class AeModule(object):
  def __init__(self, data_tensor, output_channels, decay_mult, act_func, name="AE"):
    """
    Autoencoder module
    Inputs:
      data_tensor
      output_channels - a list of channels to make, also defines number of layers
      decay_mult - weight decay multiplier
      act_func - activation function
      name
    Outputs:
      dictionary
    """
    data_ndim = len(data_tensor.get_shape().as_list())
    assert data_ndim == 2, (
      "Module requires datal_tensor to have shape [batch, num_pixels]")

    self.data_tensor = data_tensor
    self.batch_size, self.num_pixels = self.data_tensor.get_shape()

    self.name = str(name)
    self.output_channels = output_channels
    self.decay_mult = decay_mult
    self.act_func = act_func

    self.trainable_variables = TrainableVariableDict()

    self.num_encoder_layers = len(self.output_channels)
    self.num_decoder_layers = self.num_encoder_layers
    self.num_layers = self.num_encoder_layers + self.num_decoder_layers

    w_enc_shape = []
    w_dec_shape = []
    prev_input_features = self.num_pixels
    for l in range(self.num_encoder_layers):
      w_enc_shape.append([prev_input_features, self.output_channels[l]])
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
      #Want to avg over batch, sum over the rest
      reduc_dim = list(range(1, len(reconstruction.shape)))
      recon_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(tf.square(tf.subtract(reconstruction, self.data_tensor)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def layer_maker(self, layer_id, input_tensor, activation_function, w_shape):
    """
    Make layer that does act(u*w+b)
    """
    with tf.variable_scope("layer"+str(layer_id), reuse=tf.AUTO_REUSE) as scope:
      w_name = "w_"+str(layer_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      b_name = "b_"+str(layer_id)
      b = tf.get_variable(name=b_name, shape=w_shape[1],
        dtype=tf.float32, initializer=self.b_init, trainable=True)
      output_tensor = activation_function(tf.add(tf.matmul(input_tensor, w), b))
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

  def build_decoder(self, start_layer_id, input_tensor, activation_functions, w_shapes):
    assert len(activation_functions) == len(w_shapes), (
      "activation_functions & w_shapes must be the same length")
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []
    layer_id = start_layer_id
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
      self.w_init = tf.initializers.random_normal(mean=0.0, stddev=0.1, dtype=tf.float32)
      self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

    self.u_list = [self.data_tensor]
    self.w_list = []
    self.b_list = []
    enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
      [self.act_func,]*self.num_encoder_layers, self.w_shapes[:self.num_encoder_layers])
    self.u_list += enc_u_list
    self.w_list += enc_w_list
    self.b_list += enc_b_list

    dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.num_encoder_layers+1,
      self.u_list[-1], [self.act_func,]*(self.num_decoder_layers-1) + [tf.identity],
      self.w_shapes[self.num_encoder_layers:])
    self.u_list += dec_u_list
    self.w_list += dec_w_list
    self.b_list += dec_b_list

    for w,b in zip(self.w_list, self.b_list):
      self.trainable_variables[w.name] = w
      self.trainable_variables[b.name] = b

    with tf.variable_scope("inference") as scope:
      self.a = tf.identity(self.u_list[int(self.num_layers/2-1)], name="activations")

    with tf.name_scope("output") as scope:
      self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

    with tf.name_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
