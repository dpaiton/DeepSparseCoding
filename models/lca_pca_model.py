import tensorflow as tf
import params.param_picker as pp
from models.lca_model import LcaModel

class LcaPcaModel(LcaModel):
  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    super(LcaPcaModel, self).build_graph_from_input(input_node)

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("pooling_filters") as scope:
          self.full_cov = tf.compat.v1.placeholder(tf.float32,
            shape=(self.params.num_neurons, self.params.num_neurons),
            name="full_covariance_matrix")
          self.eigen_vals, self.eigen_vecs = tf.linalg.eigh(self.full_cov, name="eig_decomp")
          self.inv_sigma = tf.where(self.eigen_vals<1e-3,
            tf.linalg.tensor_diag(tf.divide(1.0,
            tf.sqrt(self.eigen_vals + self.params.eps))),
            tf.linalg.tensor_diag(tf.zeros_like(self.eigen_vals)),
            name="inv_sigma")
          top_vecs = self.eigen_vecs[:, :self.params.num_pooling_units]
          self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
            name="pooling_filters")

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("covariance") as scope:
          act_centered = self.get_encodings() - tf.reduce_mean(self.get_encodings(),
            axis=[1], keepdims=True)
          self.act_cov = tf.divide(tf.matmul(tf.transpose(act_centered), act_centered),
            tf.cast(tf.shape(input_node)[0], dtype=tf.float32), name="a_cov_matrix")

        with tf.compat.v1.variable_scope("inference") as scope:
          self.a2 = tf.matmul(self.get_encodings(), self.eigen_vecs, name="a2")
          self.pooled_activity = tf.matmul(self.get_encodings(), self.pooling_filters, name="pooled_act")
