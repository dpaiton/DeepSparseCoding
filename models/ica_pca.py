import numpy as np
import tensorflow as tf
import params.param_picker as pp
from models.ica import ICA

class ICA_PCA(ICA):
  def __init__(self, params, schedule):
    ica_params, ica_schedule = pp.get_params("ica")
    new_params = ica_params.copy()
    new_params.update(params)
    super(ICA_PCA, self).__init__(new_params, schedule)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    num_pooling_units [int] indicating the number of 2nd layer units
  """
  def load_params(self, params):
    super(ICA_PCA, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])

  """Build the TensorFlow graph object"""
  def build_graph(self):
    super(ICA_PCA, self).build_graph()
    with self.graph.as_default():
      with tf.variable_scope("weights") as scope:
        self.phi = tf.transpose(self.a)

      with tf.name_scope("covariance") as scope:
        self.act_corr = tf.divide(tf.matmul(tf.transpose(tf.nn.relu(self.u)),
          tf.nn.relu(self.u)), tf.to_float(tf.shape(self.x)[0]), name="a_corr_matrix")
        act_centered = tf.nn.relu(self.u) - tf.reduce_mean(tf.nn.relu(self.u), axis=[1],
          keep_dims=True)
        self.act_cov = tf.divide(tf.matmul(tf.transpose(act_centered), act_centered),
          tf.to_float(tf.shape(self.x)[0]), name="a_cov_matrix")

      with tf.name_scope("pooling_filters") as scope:
        self.full_cov = tf.placeholder(tf.float32, shape=(self.num_neurons, self.num_neurons),
          name="full_covariance_matrix")
        s, u, v = tf.svd(self.full_cov, full_matrices=True, name="a_svd")
        top_vecs = u[:, :self.num_pooling_units]
        self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
          name="pooling_filters")
