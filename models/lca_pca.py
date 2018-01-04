import tensorflow as tf
import params.param_picker as pp
from models.lca import LCA

class LCA_PCA(LCA):
  def setup(self, params, schedule):
    lca_params, lca_schedule = pp.get_params("lca")
    new_params = lca_params.copy()
    new_params.update(params)
    super(LCA_PCA, self).setup(new_params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      num_pooling_units [int] indicating the number of 2nd layer units
    """
    super(LCA_PCA, self).load_params(params)
    self.num_pooling_units = int(params["num_pooling_units"])

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with self.graph.as_default():
      with tf.name_scope("pooling_filters") as scope:
        self.full_cov = tf.placeholder(tf.float32, shape=(self.num_neurons, self.num_neurons),
          name="full_covariance_matrix")
        self.eigen_vals, self.eigen_vecs = tf.self_adjoint_eig(self.full_cov, name="eig_decomp")
        self.inv_sigma = tf.where(self.eigen_vals<1e-3, tf.diag(tf.divide(1.0,
          tf.sqrt(self.eigen_vals + self.eps))), tf.diag(tf.zeros_like(self.eigen_vals)),
          name="inv_sigma")
        top_vecs = self.eigen_vecs[:, :self.num_pooling_units]
        self.pooling_filters = tf.transpose(tf.matmul(top_vecs, tf.transpose(top_vecs)),
          name="pooling_filters")

    super(LCA_PCA, self).build_graph()

    with self.graph.as_default():
      with tf.name_scope("covariance") as scope:
        act_centered = self.a - tf.reduce_mean(self.a, axis=[1], keep_dims=True)
        self.act_cov = tf.divide(tf.matmul(tf.transpose(act_centered), act_centered),
          tf.to_float(tf.shape(self.x)[0]), name="a_cov_matrix")

      with tf.variable_scope("inference") as scope:
        self.a2 = tf.matmul(self.a, self.eigen_vecs, name="a2")
        self.pooled_activity = tf.matmul(self.a, self.pooling_filters, name="pooled_act")
