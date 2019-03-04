import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class AeModule(object):
  #TODO consolidate patch_size_y and patch_size_x into one variable
  #TODO propogate above change to all other modules
  def __init__(self, data_tensor, layer_types, output_channels, patch_size_y,
    patch_size_x, conv_strides, decay_mult, act_funcs, dropout,
    tie_decoder_weights, variable_scope="ae"):
    """
    Autoencoder module
    Inputs:
      data_tensor
      output_channels: a list of channels to make, also defines number of layers
      decay_mult: weight decay multiplier
      act_funcs: activation functions
      dropout: specifies the keep probability or None
      variable_scope: specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.variable_scope = variable_scope

    data_ndim = len(data_tensor.get_shape().as_list())
    if(layer_types[0] == "conv"):
      assert data_ndim == 4, (
        "Module requires data_tensor to have shape [batch, num_pixels_y, num_pixels_x, num_features]" +
        " if first layer is conv")
    else:
      assert data_ndim == 4 or data_ndim == 2, (
        "Module requires data_tensor to have shape [batch, num_pixels_y, num_pixels_x, num_features]" +
        " or [batch, num_pixels]")

    self.data_tensor = data_tensor

    self.layer_types = layer_types
    self.output_channels = output_channels
    self.patch_size_y = patch_size_y
    self.patch_size_x = patch_size_x
    self.conv_strides = conv_strides

    self.dropout = dropout

    self.decay_mult = decay_mult

    self.trainable_variables = TrainableVariableDict()

    self.num_fc_layers = layer_types.count("fc")
    self.num_conv_layers = layer_types.count("conv")
    self.num_encoder_layers = self.num_fc_layers + self.num_conv_layers

    self.num_encoder_layers = len(self.output_channels)

    assert len(layer_types) == self.num_encoder_layers, \
      ("All layer_types must be conv or fc")
    assert len(self.output_channels) == self.num_encoder_layers, \
      ("output_channels must be a list of size " + str(self.num_encoder_layers))
    assert len(patch_size_y) == self.num_encoder_layers, \
      ("patch_size_y must be a list of size " + str(self.num_encoder_layers))
    assert len(patch_size_x) == self.num_encoder_layers, \
      ("patch_size_x must be a list of size " + str(self.num_encoder_layers))
    assert len(conv_strides) == self.num_encoder_layers, \
      ("conv_strides must be a list of size " + str(self.num_encoder_layers))

    self.num_decoder_layers = self.num_encoder_layers
    self.num_layers = self.num_encoder_layers + self.num_decoder_layers

    self.act_funcs = act_funcs
    assert len(self.act_funcs) == self.num_layers, \
      ("act_funcs parameter must be a list of size " + str(self.num_layers))

    self.tie_decoder_weights = tie_decoder_weights

    self.build_graph()

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

  def conv_layer_maker(self, layer_id, input_tensor, activation_function, w_shape,
    strides, transpose=False, output_shape=None):
    """
    Make layer that does conv2d (or conv2d_transpose) of input tensor
    Example case for w_read_id logic:
      layer_id: [0 1 2 3 4] [5 6 7 8 9]

                              10-6  10-7  10-8 10-9  10-10
      weight_id: [0 1 2 3 4] [ 4     3     2     1     0 ]
      num_layers: 10
      weight_id = num_layers - (layer_id + 1)
    output_shape is needed if decoder layer
    """
    with tf.variable_scope("layer"+str(layer_id), reuse=tf.AUTO_REUSE) as scope:
      if self.tie_decoder_weights:
        w_read_id = self.num_layers - (layer_id+1)
      else:
        w_read_id = layer_id

      w_name = "conv_w_"+str(w_read_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)

      b_name = "conv_b_"+str(layer_id)
      if(transpose):
        b_shape = w_shape[-2]
      else:
        b_shape = w_shape[-1]

      b = tf.get_variable(name=b_name, shape=b_shape,
        dtype=tf.float32, initializer=self.b_init, trainable=True)

      if(transpose):
        assert(output_shape is not None)
        output_tensor = activation_function(
          tf.add(tf.nn.conv2d_transpose(input_tensor, w, output_shape, strides,
          padding="SAME"), b))
      else:
        output_tensor = activation_function(
          tf.add(tf.nn.conv2d(input_tensor, w, strides, padding="SAME"), b))

      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, w, b

  def fc_layer_maker(self, layer_id, input_tensor, activation_function, w_shape):
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

      w_name = "fc_w_"+str(w_read_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)

      b_name = "fc_b_"+str(layer_id)
      b = tf.get_variable(name=b_name, shape=w_shape[1],
        dtype=tf.float32, initializer=self.b_init, trainable=True)

      output_tensor = activation_function(tf.add(tf.matmul(input_tensor, w), b))
      output_tensor = tf.nn.dropout(output_tensor, keep_prob=self.dropout[layer_id])
    return output_tensor, w, b

  def build_encoder(self, input_tensor, activation_functions):
    enc_u_list = [input_tensor]
    enc_w_list = []
    enc_b_list = []

    prev_input_features = input_tensor.get_shape().as_list()[-1]
    #TODO assert all conv layers before fc layers
    for layer_id in range(self.num_conv_layers):
      w_shape = [int(self.patch_size_y[layer_id]), int(self.patch_size_x[layer_id]),
        int(prev_input_features), int(self.output_channels[layer_id])]
      u_out, w, b = self.conv_layer_maker(layer_id, enc_u_list[layer_id], activation_functions[layer_id],
        w_shape, self.conv_strides[layer_id], transpose=False)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = self.output_channels[layer_id]

    enc_shape = enc_u_list[-1].get_shape().as_list()
    if(len(enc_shape) == 4):
      (batch, y, x, f) = enc_shape
      #Flatten last conv layer
      prev_input_features = y*x*f
      flat_fc_input = tf.reshape(enc_u_list[-1], [-1, prev_input_features])
    elif(len(enc_shape) == 2):
      flat_fc_input = enc_u_list[-1]
    else:
      assert False

    for fc_layer_id in range(self.num_fc_layers):
      layer_id = fc_layer_id + self.num_conv_layers

      w_shape = [int(prev_input_features), int(self.output_channels[layer_id])]
      if(fc_layer_id == 0):
        in_tensor = flat_fc_input
      else:
        in_tensor = enc_u_list[layer_id]

      u_out, w, b = self.fc_layer_maker(layer_id, in_tensor, activation_functions[layer_id],
        w_shape)
      enc_u_list.append(u_out)
      enc_w_list.append(w)
      enc_b_list.append(b)
      prev_input_features = int(self.output_channels[layer_id])

    return enc_u_list, enc_w_list, enc_b_list

  #enc_u_list needed for shapes
  #enc_w_list needed for infering weight shapes
  #TODO this decoder is asserting that the decoder mirrors the encoder
  #i.e., this function now accesses member variables (u_list and w_list) to
  #find weight shapes
  def build_decoder(self, input_tensor, activation_functions):
    dec_u_list = [input_tensor]
    dec_w_list = []
    dec_b_list = []

    for dec_layer_id in range(self.num_fc_layers):
      layer_id = self.num_encoder_layers + dec_layer_id
      #Corresponding enc layer
      enc_w_id = -(dec_layer_id+1)
      enc_w_shape = self.enc_w_list[enc_w_id].get_shape().as_list()
      w_shape = enc_w_shape[::-1]
      u_out, w, b = self.fc_layer_maker(layer_id, dec_u_list[dec_layer_id],
        activation_functions[dec_layer_id], w_shape)

      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)

    #Reshape flat vector for next conv
    u_list_id = -(self.num_fc_layers + 1)
    enc_shape = self.enc_u_list[u_list_id].get_shape().as_list()
    if(len(enc_shape) == 4):
      (batch, y, x, f) = self.enc_u_list[u_list_id].get_shape().as_list()
      reshape_conv_input = tf.reshape(dec_u_list[-1], [-1, y, x, f])

    for dec_conv_layer_id in range(self.num_conv_layers):
      dec_layer_id = self.num_fc_layers + dec_conv_layer_id
      layer_id = self.num_encoder_layers + dec_layer_id
      enc_w_id = -(dec_layer_id + 1)
      w_shape = self.enc_w_list[enc_w_id].get_shape().as_list()

      #u_list_id is the id for the INPUT of this layer
      u_list_id = -(dec_layer_id + 1)
      #-1 more since we want the shape of the output
      batch_size = tf.shape(self.enc_u_list[u_list_id-1])[0]
      act_size = self.enc_u_list[u_list_id-1].get_shape().as_list()[1:]
      output_shape = tf.stack([batch_size, ] + act_size)

      if(dec_conv_layer_id == 0):
        in_tensor = reshape_conv_input
      else:
        in_tensor = dec_u_list[dec_layer_id]
      u_out, w, b = self.conv_layer_maker(layer_id, in_tensor,
        activation_functions[dec_conv_layer_id], w_shape,
        self.conv_strides[u_list_id], transpose=True, output_shape=output_shape)

      dec_u_list.append(u_out)
      dec_w_list.append(w)
      dec_b_list.append(b)

    return dec_u_list, dec_w_list, dec_b_list

  def compute_total_loss(self):
    with tf.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "weight_decay_loss":self.compute_weight_decay_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")

  def build_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope("weight_inits") as scope:
        self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01, dtype=tf.float32)
        self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

      self.u_list = [self.data_tensor]
      self.w_list = []
      self.b_list = []

      enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
        self.act_funcs[:self.num_encoder_layers])
      #Need these member variables for build_decoder
      self.enc_u_list = enc_u_list
      self.enc_w_list = enc_w_list

      self.u_list += enc_u_list
      self.w_list += enc_w_list
      self.b_list += enc_b_list

      with tf.variable_scope("inference") as scope:
        self.a = tf.identity(enc_u_list[-1], name="activity")

      dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.u_list[-1],
        self.act_funcs[self.num_encoder_layers:])

      self.u_list += dec_u_list
      if not self.tie_decoder_weights:
        self.w_list += dec_w_list
      self.b_list += dec_b_list

      for w,b in zip(self.w_list, self.b_list):
        self.trainable_variables[w.name] = w
        self.trainable_variables[b.name] = b

      with tf.variable_scope("output") as scope:
        self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

      self.compute_total_loss()

