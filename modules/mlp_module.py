import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.batch_normalization_module import BatchNormalizationModule

class MlpModule(object):
  def __init__(self, data_tensor, label_tensor, layer_types, output_channels, batch_norm,
      dropout, max_pool, max_pool_ksize, max_pool_strides, patch_size_y, patch_size_x,
      conv_strides, eps, loss_type="softmax_cross_entropy", name_scope="MLP"):
    """
    Multi Layer Perceptron module for 1-hot labels
    Inputs:
      data_tensor
      label_tensor
      layer_types list of "conv" or "fc"
      output_channels is a list of ints
      patch_size_(y,x) specifies the patch size in tye (y,x) directions
      conv_strides specifies strides in (batch, y, x, channels) directions
      max_pool is a list of booleans
      dropout specifies the keep probability or None
      batch_norm is a list of decay multipliers or None
      eps is a float
      loss_type is a string specifying the type of loss ("softmax_cross_entropy" or "l2")
      name_scope is a string
    Outputs:
      dictionary
    TODO: relu is hard coded, but should be a parameter that is passed to an activation module
      that has a bunch of activations
    """
    data_shape = data_tensor.get_shape().as_list()

    data_ndim = len(data_shape)
    assert (data_ndim == 2 or data_ndim == 4), (
      "Model requires data_tensor to have shape [batch, num_features] or [batch, y, x, features]")
    label_ndim = len(label_tensor.get_shape().as_list())
    assert label_ndim == 2, (
      "Model requires label_tensor to have shape [batch, num_classes]")

    self.data_tensor = data_tensor
    if data_ndim == 2:
      self.batch_size, self.num_pixels = data_shape
      assert layer_types[0] == "fc", ("Data tensor must have ndim==2 for fc layers")
    elif data_ndim == 4:
      self.batch_size, self.y_size, self.x_size, self.num_data_channels = data_shape
      self.num_pixels = self.y_size * self.x_size * self.num_data_channels
      assert layer_types[0] == "conv", ("Data tensor must have ndim==4 for conv layers")
    else:
      assert False, ("Shouldn't get here")

    self.loss_type = loss_type
    assert self.loss_type == "softmax_cross_entropy" or self.loss_type == "l2",\
      ("Acceptable loss functions are \"softmax_cross_entropy\" or \"l2\"")

    self.label_tensor = label_tensor
    label_batch, self.num_classes = label_tensor.get_shape()

    # load params
    self.layer_types = layer_types
    self.max_pool = max_pool
    self.max_pool_ksize = max_pool_ksize
    self.max_pool_strides = max_pool_strides
    self.dropout = dropout
    self.batch_norm = batch_norm
    # assert no FC after Conv in layer types
    self.output_channels = output_channels
    self.patch_size_y = patch_size_y
    self.patch_size_x = patch_size_x
    self.conv_strides = conv_strides
    self.eps = eps
    self.name_scope = name_scope

    # computed params
    self.num_fc_layers = layer_types.count("fc")
    self.num_conv_layers = layer_types.count("conv")
    self.num_layers = self.num_fc_layers + self.num_conv_layers

    #Check that layer definition parameters are of the same length
    assert len(layer_types) == self.num_layers, \
      ("All layer_types must be conv or fc")
    assert len(output_channels) == self.num_layers, \
      ("output_channels must be a list of size " + str(self.num_layers))
    assert len(patch_size_y) == self.num_layers, \
      ("patch_size_y must be a list of size " + str(self.num_layers))
    assert len(patch_size_x) == self.num_layers, \
      ("patch_size_x must be a list of size " + str(self.num_layers))
    assert len(conv_strides) == self.num_layers, \
      ("conv_strides must be a list of size " + str(self.num_layers))
    assert len(batch_norm) == self.num_layers, \
      ("batch_norm must be a list of size " + str(self.num_layers))
    assert len(max_pool) == self.num_layers, \
      ("max_pool must be a list of size " + str(self.num_layers))
    assert len(max_pool_ksize) == self.num_layers, \
      ("max_pool_ksize must be a list of size " + str(self.num_layers))
    assert len(max_pool_strides) == self.num_layers, \
      ("max_pool_strides must be a list of size " + str(self.num_layers))

    if(data_ndim == 4): #If at least one convolutional layers
      conv_input_channels = [self.num_data_channels] + self.output_channels[:-1]
      self.conv_w_shapes = [vals for vals in zip(self.patch_size_y, self.patch_size_x,
        conv_input_channels, self.output_channels)]

    self.fc_output_channels = self.output_channels[self.num_conv_layers:]

    self.trainable_variables = TrainableVariableDict()
    self.build_graph()

  def conv_layer_maker(self, layer_id, a_in, w_shape, strides, b_shape):
    with tf.variable_scope("layer"+str(layer_id)) as scope:
      w_name = "conv_w_"+str(layer_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      self.trainable_variables[w.name] = w

      b_name = "conv_b_"+str(layer_id)
      b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32,
        initializer=self.b_init, trainable=True)
      self.trainable_variables[b.name] = b

      conv_out = tf.nn.relu(tf.add(tf.nn.conv2d(a_in, w, strides, padding="SAME"),
        b), name="conv_out"+str(layer_id))
      if self.batch_norm[layer_id] is not None:
        bn = BatchNormalizationModule(conv_out, self.batch_norm[layer_id], self.eps,
          reduc_axes=[0,1,2], name="BatchNorm_"+str(layer_id))
        conv_out = bn.get_output()
        self.trainable_variables.update(bn.trainable_variables)
      conv_out = tf.nn.dropout(conv_out, keep_prob=self.dropout[layer_id])
      if self.max_pool[layer_id]:
        conv_out = tf.nn.max_pool(conv_out, ksize=self.max_pool_ksize[layer_id],
          strides=self.max_pool_strides[layer_id], padding="SAME")
    return conv_out, w, b

  def fc_layer_maker(self, layer_id, a_in, w_shape, b_shape, act_func):
    with tf.variable_scope("layer"+str(layer_id)) as scope:
      w_name = "fc_w_"+str(layer_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      self.trainable_variables[w.name] = w

      b_name = "fc_b_"+str(layer_id)
      b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32,
        initializer=self.b_init, trainable=True)
      self.trainable_variables[b.name] = b

      fc_out = act_func(tf.add(tf.matmul(a_in, w), b), name="fc_out"+str(layer_id))
      if self.batch_norm[layer_id] is not None:
        bn = BatchNormalizationModule(fc_out, self.batch_norm[layer_id], self.eps, reduc_axes=[0],
          name="BatchNorm_"+str(layer_id))
        fc_out = bn.get_output()
        self.trainable_variables.update(bn.trainable_variables)
      fc_out = tf.nn.dropout(fc_out, keep_prob=self.dropout[layer_id])
      if self.max_pool[layer_id]:
        fc_out = tf.nn.max_pool(fc_out, ksize=self.max_pool_ksize[layer_id],
          strides=self.max_pool_strides[layer_id], padding="SAME")
    return fc_out, w, b

  def make_layers(self):
    act_list = [self.data_tensor]
    w_list = []
    b_list = []
    for layer_id in range(self.num_conv_layers):
      a_out, w, b = self.conv_layer_maker(layer_id, act_list[layer_id],
      self.conv_w_shapes[layer_id], self.conv_strides[layer_id], self.output_channels[layer_id])
      act_list.append(a_out)
      w_list.append(w)
      b_list.append(b)
    #TODO: Make this a parameter like in the auto-encoder
    act_funcs = [tf.nn.relu,]*(self.num_fc_layers-1) + [tf.identity]
    for fc_layer_id in range(self.num_fc_layers):
      layer_id = fc_layer_id + self.num_conv_layers
      a_resh = tf.contrib.layers.flatten(act_list[layer_id])
      w_shape = [a_resh.get_shape()[1].value, self.fc_output_channels[fc_layer_id]]
      a_out, w, b = self.fc_layer_maker(layer_id, a_resh, w_shape, self.output_channels[layer_id],
        act_funcs[fc_layer_id])
      act_list.append(a_out)
      w_list.append(w)
      b_list.append(b)
    return act_list, w_list, b_list

  def build_graph(self):
    """
    Build an MLP TensorFlow Graph.
    """
    with tf.name_scope(self.name_scope) as scope:
      with tf.name_scope("weight_inits") as scope:
        self.w_init = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        self.b_init = tf.initializers.constant(0.1, dtype=tf.float32)

      self.layer_list, self.weight_list, self.bias_list = self.make_layers()

      with tf.name_scope("output") as scope:
        with tf.name_scope("label_estimate"):
          self.label_est = tf.nn.softmax(self.layer_list[-1])

      with tf.name_scope("loss") as scope:
        with tf.name_scope("supervised"):
          with tf.name_scope(self.loss_type):
            if(self.loss_type == "softmax_cross_entropy"):
              ## For semi-supervised learning, loss is 0 if there is no label
              self.label_mult = tf.reduce_sum(self.label_tensor, axis=[1]) # vector with len [batch]
              self.cross_entropy_loss = (self.label_mult
                * -tf.reduce_sum(tf.multiply(self.label_tensor, tf.log(tf.clip_by_value(
                self.label_est, self.eps, 1.0))), axis=[1])) # vector with len [batch]
              #Doing this to avoid divide by zero
              label_count = tf.reduce_sum(self.label_mult) # number of non-ignore labels in batch
              f1 = lambda: tf.zeros_like(self.cross_entropy_loss)
              f2 = lambda: tf.reduce_sum(self.cross_entropy_loss) / label_count
              pred_fn_pairs = {
                tf.equal(label_count, tf.constant(0.0)): f1, # all labels are 'ignore'
                tf.greater(label_count, tf.constant(0.0)): f2} # mean over non-ignore labels
              self.sum_loss = self.cross_entropy_loss
              self.mean_loss = tf.case(pred_fn_pairs,
                default=f2, exclusive=True, name="mean_cross_entropy_loss")
            elif(self.loss_type == "l2"):
              #TODO allow for semi-supervised learning with l2 loss
              # Want to avg over batch, sum over the rest
              reduc_dim = list(range(1, len(self.label_tensor.shape)))
              # Label_tensor sometimes can depend on trainable variables
              labels = tf.stop_gradient(self.label_tensor)
              self.l2_loss = tf.reduce_sum(tf.square(labels - self.layer_list[-1]), axis=reduc_dim)
              self.sum_loss = tf.reduce_sum(self.l2_loss)
              self.mean_loss = tf.reduce_mean(self.l2_loss)
          self.supervised_loss = self.mean_loss
        self.total_loss = self.supervised_loss
