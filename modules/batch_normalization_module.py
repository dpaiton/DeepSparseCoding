import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class BatchNormalizationModule(object):
  """
  Implements batch normalization
  Inputs:
    a_in [tensor] feature map (or vector) for layer
    norm_decay_mult [tf.placeholder] decay multiplier
    reduc_axes [list of ints] what axes to reduce over; default is for fc layer,
      [0,1,2] should be used for a conv layer
  """
  def __init__(self, a_in, norm_decay_mult, eps, reduc_axes, variable_scope="batch_norm"):
    self.trainable_variables = TrainableVariableDict()

    with tf.variable_scope(variable_scope) as scope:
      bn_scale_name = "scale_var"
      batch_norm_scale = tf.get_variable(name=bn_scale_name,
        dtype=tf.float32, initializer=tf.constant(1.0))
      self.trainable_variables[batch_norm_scale.name] = batch_norm_scale

      bn_shift_name = "shift_var"
      batch_norm_shift = tf.get_variable(name=bn_shift_name,
        dtype=tf.float32, initializer=tf.constant(0.0))
      self.trainable_variables[batch_norm_shift.name] = batch_norm_shift

      num_layer_features = a_in.get_shape()[-1]
      layer_means = tf.Variable(tf.zeros([num_layer_features]),
        dtype=tf.float32, trainable=False)
      layer_vars = tf.Variable(0.01*tf.ones([num_layer_features]),
        dtype=tf.float32, trainable=False)

      input_mean, input_var = tf.nn.moments(a_in, axes=reduc_axes)
      layer_means = ((1 - norm_decay_mult) * layer_means + norm_decay_mult * input_mean)
      layer_vars = ((1 - norm_decay_mult) * layer_vars + norm_decay_mult * input_var)
      adj_a_in = tf.divide(tf.subtract(a_in, layer_means), tf.sqrt(tf.add(layer_vars, eps)))

      self.act_out = tf.add(tf.multiply(batch_norm_scale, adj_a_in), batch_norm_shift)

  def get_output(self):
    return self.act_out

