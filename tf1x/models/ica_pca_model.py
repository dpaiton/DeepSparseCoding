import tensorflow as tf

import DeepSparseCoding.tf1x.params.param_picker as pp
from DeepSparseCoding.tf1x.models.ica_model import IcaModel

class IcaPcaModel(IcaModel):
  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    super(IcaPcaModel, self).build_graph_from_input(input_node)
    with self.graph.as_default():
      with tf.compat.v1.variable_scope("covariance") as scope:
        self.act_corr = tf.math.divide(tf.matmul(tf.transpose(tf.nn.relu(self.a)),
          tf.nn.relu(self.a)), tf.to_float(tf.shape(input_node)[0]), name="a_corr_matrix")
        act_centered = tf.nn.relu(self.a) - tf.reduce_mean(tf.nn.relu(self.a), axis=[1],
          keep_dims=True)
        self.act_cov = tf.math.divide(tf.matmul(tf.transpose(act_centered), act_centered),
          tf.to_float(tf.shape(input_node)[0]), name="a_cov_matrix")

      with tf.compat.v1.variable_scope("pooling_filters") as scope:
        self.full_cov = tf.compat.v1.placeholder(tf.float32, shape=(self.num_neurons, self.num_neurons),
          name="full_covariance_matrix")
        s, u, v = tf.svd(self.full_cov, full_matrices=True, name="a_svd")
        top_vecs = u[:, :self.params.num_pooling_units]
        self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
          name="pooling_filters")
