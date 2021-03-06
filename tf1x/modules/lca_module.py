import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.utils.trainable_variable_dict import TrainableVariableDict
from DeepSparseCoding.tf1x.modules.activations import lca_threshold

class LcaModule(object):
  def __init__(self, data_tensor, num_neurons, sparse_mult, step_size, thresh_type,
      rectify_a, num_steps, eps, variable_scope="lca"):
    """
    Locally Competitive Algorithm module
    Inputs:
      data_tensor
      num_neurons
      sparse_mult
      step_size
      thresh_type
      rectify_a
      num_steps
      eps
      variable_scope: specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.data_tensor = data_tensor
    self.check_data()
    self.variable_scope = variable_scope
    self.num_neurons = num_neurons
    self.sparse_mult = sparse_mult
    self.step_size = step_size
    self.thresh_type = thresh_type
    self.rectify_a = rectify_a
    self.num_steps = num_steps
    self.eps = eps
    self.calc_shapes()
    self.trainable_variables = TrainableVariableDict()
    self.build_graph()

  def check_data(self):
    data_ndim = len(self.data_tensor.get_shape().as_list())
    assert data_ndim == 2, ("Module requires datal_tensor to have shape [batch, num_pixels]")
    self.batch_size, self.num_pixels = self.data_tensor.get_shape()

  def calc_shapes(self):
    self.w_shape = [int(self.num_pixels), self.num_neurons]
    self.u_shape = [self.num_neurons]

  def compute_excitatory_current(self):
    return tf.matmul(self.data_tensor, self.w, name="driving_input")

  def compute_inhibitory_connectivity(self):
   return (tf.matmul(tf.transpose(a=self.w), self.w, name="gram_matrix")
     - tf.constant(np.identity(self.w_shape[1], dtype=np.float32), name="identity_matrix"))

  def threshold_units(self, u_in, name=None):
    return lca_threshold(u_in, self.thresh_type, self.rectify_a, self.sparse_mult, name)

  def step_inference(self, u_in, a_in, b, g, step):
    with tf.compat.v1.variable_scope("update_u"+str(step)) as scope:
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      du = tf.subtract(tf.subtract(b, lca_explain_away), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.step_size, du))
    return u_out, lca_explain_away

  def infer_coefficients(self):
   lca_b = self.compute_excitatory_current()
   lca_g = self.compute_inhibitory_connectivity()
   u_list = [self.u_zeros]
   a_list = [self.threshold_units(u_list[0])]
   for step in range(self.num_steps-1):
     u = self.step_inference(u_list[step], a_list[step], lca_b, lca_g, step)[0]
     u_list.append(u)
     a_list.append(self.threshold_units(u))
   return (u_list, a_list)

  def build_decoder(self, input_tensor, name=None):
    return tf.matmul(input_tensor, tf.transpose(a=self.w), name=name)

  def compute_recon_loss(self, recon):
    with tf.compat.v1.variable_scope("unsupervised"):
      reduc_dim = list(range(1, len(recon.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(input_tensor=0.5 *
        tf.reduce_sum(input_tensor=tf.square(tf.subtract(self.data_tensor, recon)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.compat.v1.variable_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      sparse_loss = self.sparse_mult * tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.abs(a_in),
        axis=reduc_dim), name="sparse_loss")
    return sparse_loss

  def build_graph(self):
    with tf.compat.v1.variable_scope(self.variable_scope) as scope:
      with tf.compat.v1.variable_scope("constants") as scope:
        u_full_shape = tf.stack([tf.shape(input=self.data_tensor)[0]]+self.u_shape)
        self.u_zeros = tf.zeros(shape=u_full_shape, dtype=tf.float32, name="u_zeros")
        #self.u_noise = tf.random.truncated_normal(shape=u_full_shape, mean=0.0, stddev=0.1,
        #  dtype=tf.float32, name="u_noise")

      w_norm_dim = list(range(len(self.w_shape)-1)) # normalize across input dim(s)
      with tf.compat.v1.variable_scope("weights") as scope:
        self.weight_scope = tf.compat.v1.get_variable_scope()
        w_init = tf.nn.l2_normalize(tf.random.truncated_normal(self.w_shape, mean=0.0,
          stddev=0.5, dtype=tf.float32), axis=w_norm_dim, epsilon=self.eps, name="w_init")
        self.w = tf.compat.v1.get_variable(name="w", dtype=tf.float32, initializer=w_init,
          trainable=True)
        self.trainable_variables[self.w.name] = self.w

      with tf.compat.v1.variable_scope("norm_weights") as scope:
        self.norm_w = self.w.assign(tf.nn.l2_normalize(self.w, axis=w_norm_dim,
          epsilon=self.eps, name="row_l2_norm"))

      with tf.compat.v1.variable_scope("inference") as scope:
        self.inference_scope = tf.compat.v1.get_variable_scope()
        u_list, a_list = self.infer_coefficients()
        self.u = tf.identity(u_list[-1], name="u")
        self.a = tf.identity(a_list[-1], name="activity")

      with tf.compat.v1.variable_scope("output") as scope:
        self.reconstruction = self.build_decoder(self.a, name="reconstruction")

      with tf.compat.v1.variable_scope("loss") as scope:
        self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
          "sparse_loss":self.compute_sparse_loss(self.a)}
        self.total_loss = tf.add_n([val for val in self.loss_dict.values()], name="total_loss")
