import tensorflow as tf

import DeepSparseCoding.tf1x.params.param_picker as pp
from DeepSparseCoding.tf1x.models.ica_model import IcaModel

class IcaPcaModel(IcaModel):
  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    super(IcaPcaModel, self).build_graph_from_input(input_node)
    with self.graph.as_default():
      with tf.compat.v1.variable_scope("covariance") as scope:
        self.act_corr = tf.math.divide(tf.matmul(tf.transpose(a=tf.nn.relu(self.a)),
          tf.nn.relu(self.a)), tf.cast(tf.shape(input=input_node)[0], dtype=tf.float32), name="a_corr_matrix")
        act_centered = tf.nn.relu(self.a) - tf.reduce_mean(input_tensor=tf.nn.relu(self.a), axis=[1],
          keepdims=True)
        self.act_cov = tf.math.divide(tf.matmul(tf.transpose(a=act_centered), act_centered),
          tf.cast(tf.shape(input=input_node)[0], dtype=tf.float32), name="a_cov_matrix")

      with tf.compat.v1.variable_scope("pooling_filters") as scope:
        self.full_cov = tf.compat.v1.placeholder(tf.float32, shape=(self.num_neurons, self.num_neurons),
          name="full_covariance_matrix")
        s, u, v = tf.linalg.svd(self.full_cov, full_matrices=True, name="a_svd")
        top_vecs = u[:, :self.params.num_pooling_units]
        self.pooling_filters = tf.transpose(a=tf.matmul(top_vecs, tf.transpose(a=top_vecs)),
          name="pooling_filters")
