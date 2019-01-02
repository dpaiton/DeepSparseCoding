import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict

class LCAModule(object):
  def __init__(self, data_tensor, num_neurons, sparse_mult, step_size, thresh_type,
      rectify_a, num_steps, eps, name="LCA"):
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
      name
    Outputs:
      dictionary
    """

    self.data_tensor = data_tensor
    self.check_data()

    self.name = str(name)
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

  def calc_shapes(self):
    self.w_shape = [int(self.num_pixels), self.num_neurons]
    self.u_shape = [self.num_neurons]

  def check_data(self):
    data_ndim = len(self.data_tensor.get_shape().as_list())
    assert data_ndim == 2, (
      "Module requires datal_tensor to have shape [batch, num_pixels]")

    if data_ndim == 2:
      self.batch_size, self.num_pixels = self.data_tensor.get_shape()
    else:
      assert False, ("Shouldn't get here")

  def compute_excitatory_current(self):
    return tf.matmul(self.data_tensor, self.w, name="driving_input")

  def compute_inhibitory_connectivity(self):
   return (tf.matmul(tf.transpose(self.w), self.w, name="gram_matrix")
     - tf.constant(np.identity(self.w_shape[1], dtype=np.float32), name="identity_matrix"))

  def threshold_units(self, u_in):
    if self.thresh_type == "soft":
      if self.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult), self.u_zeros)
      else:
        a_out = tf.where(tf.greater_equal(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult),
          tf.where(tf.less_equal(u_in, -self.sparse_mult),
          tf.add(u_in, self.sparse_mult),
          self.u_zeros))
    elif self.thresh_type == "hard":
      if self.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in, self.u_zeros)
      else:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in,
          tf.where(tf.less(u_in, -self.sparse_mult), u_in, self.u_zeros))
    else:
      a_out = tf.identity(u_in)
    return a_out

  def step_inference(self, u_in, a_in, b, g, step):
    with tf.name_scope("update_u"+str(step)) as scope:
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
     a_list.append(self.threshold_units(u_list[step+1]))
   return (u_list, a_list)

  def compute_recon(self, a_in):
    return tf.matmul(a_in, tf.transpose(self.w), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.data_tensor, self.reconstruction)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      sparse_loss = self.sparse_mult * tf.reduce_mean(tf.reduce_sum(tf.abs(a_in),
        axis=reduc_dim), name="sparse_loss")
    return sparse_loss

  def compute_total_loss(self, a_in, loss_funcs):
    """
    Returns sum of all loss functions defined in loss_funcs for given a_in
    Inputs:
      a_in [tf.Variable] containing the sparse coding activity values
      loss_funcs [dict] containing keys that correspond to names of loss functions and values that
        point to the functions themselves
    """
    total_loss = tf.add_n([func(a_in) for func in loss_funcs.values()], name="total_loss")
    return total_loss

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss, "sparse_loss":self.compute_sparse_loss}

  def build_graph(self):
    with tf.name_scope("constants") as scope:
      u_full_shape = tf.stack([tf.shape(self.data_tensor)[0]]+self.u_shape)
      self.u_zeros = tf.zeros(shape=u_full_shape, dtype=tf.float32, name="u_zeros")
      self.u_noise = tf.truncated_normal(shape=u_full_shape, mean=0.0, stddev=0.1,
        dtype=tf.float32, name="u_noise")

    w_norm_dim = list(range(len(self.w_shape)-1)) # normalize across input dim(s)

    with tf.variable_scope("weights") as scope:
      self.weight_scope = tf.get_variable_scope()
      w_init = tf.nn.l2_normalize(tf.truncated_normal(self.w_shape, mean=0.0,
        stddev=0.5, dtype=tf.float32), axis=w_norm_dim, epsilon=self.eps, name="w_init")
      self.w = tf.get_variable(name="w", dtype=tf.float32, initializer=w_init,
        trainable=True)
      self.trainable_variables[self.w.name] = self.w

    with tf.name_scope("norm_weights") as scope:
      self.norm_w = self.w.assign(tf.nn.l2_normalize(self.w, axis=w_norm_dim,
        epsilon=self.eps, name="row_l2_norm"))
      self.norm_weights = tf.group(self.norm_w, name="l2_normalization")

    with tf.variable_scope("inference") as scope:
      self.inference_scope = tf.get_variable_scope()
      u_list, a_list = self.infer_coefficients()
      self.u = tf.identity(u_list[-1], name="u")
      self.a = tf.identity(a_list[-1], name="activity")

    with tf.name_scope("output") as scope:
      self.reconstruction = self.compute_recon(self.a)

    with tf.name_scope("loss") as scope:
      loss_funcs = self.get_loss_funcs()
      self.loss_dict = dict(zip(
        [key for key in loss_funcs.keys()], [func(self.a) for func in loss_funcs.values()]))
      self.total_loss = self.compute_total_loss(self.a, loss_funcs)
