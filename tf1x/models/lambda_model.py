import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.models.base_model import Model

class LambdaModel(Model):
  def __init__(self):
    super(LambdaModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(LambdaModel, self).load_params(params)
    self.input_shape = [None,] + self.params.data_shape
    self.data_dim = np.prod(self.params.data_shape)
    if self.params.activation_function is None:
      self.params.activation_function = tf.identity
    if not hasattr(self.params, 'num_neurons'):
      self.params.num_neurons = self.data_dim

  def build_graph_from_input(self, input_node):
    """
    Build a passthrough model with anonymous activation function
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("placeholders") as scope:
          if self.params.num_neurons == self.data_dim:
            weight_init = np.identity(self.data_dim, dtype=np.float32)
          else:
            weight_init = np.ones([self.data_dim, self.params.num_neurons], dtype=np.float32)
          self.weight_placeholder = tf.compat.v1.placeholder_with_default(weight_init,
            shape=[self.data_dim, self.params.num_neurons], name="weights")

        with tf.compat.v1.variable_scope("inference") as scope:
          self.a = self.params.activation_function(tf.matmul(input_node, self.weight_placeholder,
            name="activity"))

  def get_input_shape(self):
    return self.input_shape

  # TODO: this will only work on a session call - not the same as other models
  def get_num_latent(self):
    return tf.get_shape(self.a)

  def get_encodings(self):
    return self.a

  def get_total_loss(self):
    return None
