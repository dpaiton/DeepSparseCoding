import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class SMT_POOL(Model):
  """
  Sparse Manifold Transform pooling method
  """
  def __init__(self, params, schedule):
    super(SMT_POOL, self).__init__(params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
    """
    super(SMT_POOL, self).load_params(params)
    self.batch_size = int(params["batch_size"])
    self.num_input_neurons = int(params["num_input_neurons"])
    self.num_output_neurons = int(params["num_output_neurons"])
    self.num_time_steps = int(params["num_time_steps"])
    self.in_shape = [self.batch_size, self.num_time_steps, self.num_input_neurons]
    self.pool_shape = [self.num_input_neurons, self.num_output_neurons]

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.input_activity = tf.placeholder(tf.float32, shape=self.in_shape, name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          pool_init = tf.truncated_normal(self.pool_shape, mean=0.0, stddev=0.5,
            dtype=tf.float32, name="pool_init")
          self.pool_weights = tf.get_variable(name="pool_weights", dtype=tf.float32,
            initializer=pool_init, trainable=True)

        with tf.variable_scope("output") as scope:
          self.pooled_activity = tf.matmul(self.input_activity, self.pool_weights,
            name="output_activity")

        with tf.variable_scope("loss") as scope:
          act_curvature = tf.subtract(self.pooled_activity, 
          act_gram = tf.matmul(self.pooled_activity, tf.transpose(self.pooled_activity),
            name="activity_gramian")
          ballistic_loss = tf.trace(tf.matmul(self.pool_weights, act_gram
