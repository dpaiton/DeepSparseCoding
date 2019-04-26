import numpy as np
import tensorflow as tf
from models.base_model import Model

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

  def build_graph_from_input(self, input_node):
    """
    Build a passthrough model with anonymous activation function
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("placeholders") as scope:
          weight_init = np.identity(self.data_dim, dtype=np.float32)
          self.weight_placeholder = tf.placeholder_with_default(weight_init,
            shape=[self.data_dim, self.data_dim], name="weights")

        with tf.variable_scope("inference") as scope:
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
