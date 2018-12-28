import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class  batch_normalization(object):
  """
  Implements batch normalization
  Inputs:
    a_in [tensor] feature map (or vector) for layer
    norm_decay_mult [int]
    reduc_axes [list of ints] what axes to reduce over; default is for fc layer,
      [0,1,2] should be used for a conv layer
  """
  def __init__(self, a_in, norm_decay_mult, reduc_axes, name="BatchNorm"):
    self.trainable_variables = TrainableVariableDict()

    with tf.name_scope(name) as scope:
      bn_scale_name = name+"_batch_norm_scale"
      self.batch_norm_scale[layer_idx] = tf.get_variable(name=bn_scale_name,
        dtype=tf.float32, initializer=tf.constant(1.0))
      self.trainable_variables[self.batch_norm_scale[layer_idx].name] = self.batch_norm_scale[layer_idx]

      bn_shift_name = name+"_batch_norm_shift"
      self.batch_norm_shift[layer_idx] = tf.get_variable(name=bn_shift_name,
        dtype=tf.float32, initializer=tf.constant(0.0))
      self.trainable_variables[self.batch_norm_shift[layer_idx].name] = self.batch_norm_shift[layer_idx]

      self.layer_means[layer_idx] = tf.Variable(tf.zeros([num_layer_features]),
        dtype=tf.float32, trainable=False)
      self.layer_vars[layer_idx] = tf.Variable(0.01*tf.ones([num_layer_features]),
        dtype=tf.float32, trainable=False)

      input_mean, input_var = tf.nn.moments(a_in, axes=reduc_axes)
      self.layer_means[layer_id] = ((1 - norm_decay_mult) * self.layer_means[layer_id]
        + norm_decay_mult * input_mean)
      self.layer_vars[layer_id] = ((1 - norm_decay_mult) * self.layer_vars[layer_id]
        + norm_decay_mult * input_var)
      adj_a_in = tf.divide(tf.subtract(a_in, self.layer_means[layer_id]),
        tf.sqrt(tf.add(self.layer_vars[layer_id], self.params.eps)))
      self.act_out = tf.add(tf.multiply(self.batch_norm_scale[layer_id], adj_a_in),
        self.batch_norm_shift[layer_id])

  def get_output(self):
    return self.act_out

