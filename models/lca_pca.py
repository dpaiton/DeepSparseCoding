import numpy as np
import tensorflow as tf
import params.param_picker as pp
from models.lca import LCA

class LCA_PCA(LCA):
  def __init__(self, params, schedule):
    lca_params, lca_schedule = pp.get_params("lca")
    new_params = lca_params.copy()
    new_params.update(params)
    super(LCA_PCA, self).__init__(new_params, schedule)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    num_pooling_units [int] indicating the number of 2nd layer units
  """
  def load_params(self, params):
    super(LCA_PCA, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])

  """Build the TensorFlow graph object"""
  def build_graph(self):
    super(LCA_PCA, self).build_graph()
    with self.graph.as_default():
      with tf.name_scope("covariance") as scope:
        self.act_corr = tf.divide(tf.matmul(tf.transpose(self.a), self.a),
          tf.to_float(tf.shape(self.x)[0]), name="a_corr_matrix")
        act_centered = self.a - tf.reduce_mean(self.a, axis=[1], keep_dims=True)
        self.act_cov = tf.divide(tf.matmul(tf.transpose(act_centered), act_centered),
          tf.to_float(tf.shape(self.x)[0]), name="a_cov_matrix")

      with tf.name_scope("pooling_filters") as scope:
        self.full_cov = tf.placeholder(tf.float32, shape=(self.num_neurons, self.num_neurons),
          name="full_covariance_matrix")
        s, u, v = tf.svd(self.full_cov, full_matrices=True, name="a_svd")
        top_vecs = u[:, :self.num_pooling_units]
        self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
          name="pooling_filters")
