import tensorflow as tf
import params.param_picker as pp
from models.lca_model import LcaModel

class LcaPcaModel(LcaModel):
  def build_graph(self):
    """Build the TensorFlow graph object"""
    super(LcaPcaModel, self).build_graph()

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("pooling_filters") as scope:
          self.full_cov = tf.placeholder(tf.float32,
            shape=(self.params.num_neurons, self.params.num_neurons),
            name="full_covariance_matrix")
          self.eigen_vals, self.eigen_vecs = tf.self_adjoint_eig(self.full_cov, name="eig_decomp")
          self.inv_sigma = tf.where(self.eigen_vals<1e-3, tf.diag(tf.divide(1.0,
            tf.sqrt(self.eigen_vals + self.params.eps))), tf.diag(tf.zeros_like(self.eigen_vals)),
            name="inv_sigma")
          top_vecs = self.eigen_vecs[:, :self.params.num_pooling_units]
          self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
            name="pooling_filters")

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("covariance") as scope:
          act_centered = self.get_encodings() - tf.reduce_mean(self.get_encodings(),
            axis=[1], keep_dims=True)
          self.act_cov = tf.divide(tf.matmul(tf.transpose(act_centered), act_centered),
            tf.to_float(tf.shape(self.x)[0]), name="a_cov_matrix")

        with tf.variable_scope("inference") as scope:
          self.a2 = tf.matmul(self.get_encodings(), self.eigen_vecs, name="a2")
          self.pooled_activity = tf.matmul(self.get_encodings(), self.pooling_filters, name="pooled_act")
