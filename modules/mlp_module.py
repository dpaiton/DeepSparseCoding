import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
from modules.batch_normalization_module import BatchNormalizationModule

class MlpModule(object):
  def __init__(self, data_tensor, label_tensor, layer_types, output_channels, batch_norm,
      dropout, max_pool, max_pool_ksize, max_pool_strides, patch_size, conv_strides,
      act_funcs, eps, lrn=None, loss_type="softmax_cross_entropy", variable_scope="mlp",
      decay_mult=None, norm_mult=None):
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
      lrn is a float for specifying local response normalization
      loss_type is a string specifying the type of loss ("softmax_cross_entropy" or "l2")
      variable_scope is a string
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
    if(decay_mult is None):
      self.decay_mult = 0
    else:
      self.decay_mult = decay_mult
    if(norm_mult is None):
      self.norm_mult = 0
    else:
      self.norm_mult = norm_mult

    self.layer_types = layer_types
    self.max_pool = max_pool
    self.max_pool_ksize = max_pool_ksize
    self.max_pool_strides = max_pool_strides
    self.dropout = dropout
    self.batch_norm = batch_norm
    #TODO assert no FC after Conv in layer types
    self.output_channels = output_channels
    self.patch_size_y = [size[0] for size in patch_size]
    self.patch_size_x = [size[1] for size in patch_size]
    self.conv_strides = conv_strides
    self.eps = eps
    self.variable_scope = variable_scope

    self.act_funcs = act_funcs

    # computed params
    self.num_fc_layers = layer_types.count("fc")
    self.num_conv_layers = layer_types.count("conv")
    self.num_layers = self.num_fc_layers + self.num_conv_layers

    #Check that layer definition parameters are of the same length
    assert len(layer_types) == self.num_layers, \
      ("All layer_types must be conv or fc")
    assert len(output_channels) == self.num_layers, \
      ("output_channels must be a list of size " + str(self.num_layers))
    assert len(self.patch_size_y) == self.num_conv_layers, \
      ("patch_size_y must be a list of size " + str(self.num_layers))
    assert len(self.patch_size_x) == self.num_conv_layers, \
      ("patch_size_x must be a list of size " + str(self.num_layers))
    assert len(conv_strides) == self.num_conv_layers, \
      ("conv_strides must be a list of size " + str(self.num_layers))
    assert len(batch_norm) == self.num_layers, \
      ("batch_norm must be a list of size " + str(self.num_layers))
    assert len(max_pool) == self.num_layers, \
      ("max_pool must be a list of size " + str(self.num_layers))
    assert len(max_pool_ksize) == self.num_layers, \
      ("max_pool_ksize must be a list of size " + str(self.num_layers))
    assert len(max_pool_strides) == self.num_layers, \
      ("max_pool_strides must be a list of size " + str(self.num_layers))
    assert len(self.act_funcs) == self.num_layers, \
      ("act_funcs parameter must be a list of size " + str(self.num_layers))

    if(data_ndim == 4): #If at least one convolutional layers
      conv_input_channels = [self.num_data_channels] + self.output_channels[:-1]
      self.conv_w_shapes = [vals for vals in zip(self.patch_size_y, self.patch_size_x,
        conv_input_channels, self.output_channels)]

    self.fc_output_channels = self.output_channels[self.num_conv_layers:]

    self.lrn = lrn
    #Default to not using lrn
    if(self.lrn is None):
      self.lrn = [None for i in range(self.num_layers)]

    self.trainable_variables = TrainableVariableDict()

    self.conv_strides = self.conv_strides + [None]*(self.num_fc_layers*2) + self.conv_strides[::-1]

    self.build_graph()

  def conv_layer_maker(self, layer_id, a_in, w_shape, strides, b_shape, act_func):
    with tf.variable_scope("layer"+str(layer_id)) as scope:
      w_name = "conv_w_"+str(layer_id)
      w = tf.get_variable(name=w_name, shape=w_shape, dtype=tf.float32,
        initializer=self.w_init, trainable=True)
      self.trainable_variables[w.name] = w

      b_name = "conv_b_"+str(layer_id)
      b = tf.get_variable(name=b_name, shape=b_shape, dtype=tf.float32,
        initializer=self.b_init, trainable=True)
      self.trainable_variables[b.name] = b

      conv_out = act_func(tf.add(tf.nn.conv2d(a_in, w, strides, padding="SAME"),
        b, name="conv_out"+str(layer_id)))
      if self.batch_norm[layer_id] is not None:
        bn = BatchNormalizationModule(conv_out, self.batch_norm[layer_id], self.eps,
          reduc_axes=[0,1,2], variable_scope="batch_norm_"+str(layer_id))
        conv_out = bn.get_output()
        self.trainable_variables.update(bn.trainable_variables)
      conv_out = tf.nn.dropout(conv_out, rate=1-self.dropout[layer_id])

      if self.lrn[layer_id] is not None:
        if self.lrn[layer_id] == "pre":
          #TODO these parameters are hard coded for now, move these to params
          conv_out = tf.nn.lrn(conv_out, depth_radius=4, bias=1.0, alpha=0.001/9.0,
            beta=0.75, name='norm1')

      if self.max_pool[layer_id]:
        conv_out = tf.nn.max_pool(conv_out, ksize=self.max_pool_ksize[layer_id],
          strides=self.max_pool_strides[layer_id], padding="SAME")

      if self.lrn[layer_id] is not None:
        if self.lrn[layer_id] == "post":
          #TODO these parameters are hard coded for now, move these to params
          conv_out = tf.nn.lrn(conv_out, depth_radius=4, bias=1.0, alpha=0.001/9.0,
            beta=0.75, name='norm1')
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
          variable_scope="batch_norm_"+str(layer_id))
        fc_out = bn.get_output()
        self.trainable_variables.update(bn.trainable_variables)
      fc_out = tf.nn.dropout(fc_out, rate=1-self.dropout[layer_id])
      if self.max_pool[layer_id]:
        fc_out = tf.nn.max_pool(fc_out, ksize=self.max_pool_ksize[layer_id],
          strides=self.max_pool_strides[layer_id], padding="SAME")
    return fc_out, w, b

  def make_layers(self):
    act_list = [self.data_tensor]
    w_list = []
    b_list = []
    # Conv is always before FC
    for layer_id in range(self.num_conv_layers):
      a_out, w, b = self.conv_layer_maker(layer_id, act_list[layer_id],
      self.conv_w_shapes[layer_id], self.conv_strides[layer_id], self.output_channels[layer_id],
        self.act_funcs[layer_id])
      act_list.append(a_out)
      w_list.append(w)
      b_list.append(b)
    #TODO: Make this a parameter like in the auto-encoder
    #act_funcs = [tf.nn.relu,]*(self.num_fc_layers-1) + [tf.identity]
    for fc_layer_id in range(self.num_fc_layers):
      layer_id = fc_layer_id + self.num_conv_layers
      a_resh = tf.contrib.layers.flatten(act_list[layer_id])
      w_shape = [a_resh.get_shape()[1].value, self.fc_output_channels[fc_layer_id]]
      a_out, w, b = self.fc_layer_maker(layer_id, a_resh, w_shape, self.output_channels[layer_id],
        self.act_funcs[layer_id])
      act_list.append(a_out)
      w_list.append(w)
      b_list.append(b)
    return act_list, w_list, b_list

  def compute_weight_decay_loss(self):
    with tf.variable_scope("w_decay"):
      w_decay_list = [tf.reduce_sum(tf.square(w)) for w in self.weight_list]
      decay_loss = tf.multiply(0.5*self.decay_mult, tf.add_n(w_decay_list))
    return decay_loss

  def compute_weight_norm_loss(self):
    with tf.variable_scope("w_norm"):
      w_norm_list = []
      for w in self.weight_list:
        reduc_axis = np.arange(1, len(w.get_shape().as_list()))
        w_norm = tf.reduce_sum(tf.square(1 - tf.reduce_sum(tf.square(w), axis=reduc_axis)))
        w_norm_list.append(w_norm)
      norm_loss = tf.multiply(0.5*self.norm_mult, tf.add_n(w_norm_list))
    return norm_loss

  def build_graph(self):
    """
    Build an MLP TensorFlow Graph.
    """
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope("weight_inits") as scope:
        self.w_init = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
        self.b_init = tf.initializers.constant(0.1, dtype=tf.float32)

      self.layer_list, self.weight_list, self.bias_list = self.make_layers()

      with tf.variable_scope("output") as scope:
        with tf.variable_scope("label_estimate"):
          self.label_est = tf.nn.softmax(self.layer_list[-1])

      with tf.variable_scope("loss") as scope:
        with tf.variable_scope("supervised"):
          with tf.variable_scope(self.loss_type):
            if(self.loss_type == "softmax_cross_entropy"):
              ## For semi-supervised learning, loss is 0 if there is no label
              self.label_mult = tf.reduce_sum(self.label_tensor, axis=[1]) # vector with len [batch
              label_classes = tf.argmax(self.label_tensor, axis=-1)

              self.cross_entropy_loss = self.label_mult * \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_classes,
                logits=self.layer_list[-1])

              #self.cross_entropy_loss = (self.label_mult
              #  * -tf.reduce_sum(tf.multiply(self.label_tensor, tf.log(tf.clip_by_value(
              #  self.label_est, self.eps, 1.0))), axis=[1])) # vector with len [batch]

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
        self.total_loss = self.supervised_loss + self.compute_weight_decay_loss() + \
          self.compute_weight_norm_loss()

