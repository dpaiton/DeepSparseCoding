import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
import pdb

class OmpModule(object):
  def __init__(self, data_tensor, num_neurons, num_k, eps,
      variable_scope="omp"):
    """
    Orthogonal Matching Pursuit module
    Inputs:
      data_tensor
      num_neurons
      num_k
      eps
      variable_scope: specifies the variable_scope for the module
    Outputs:
      dictionary
    """

    self.data_tensor = data_tensor
    self.check_data()

    self.variable_scope = variable_scope
    self.num_neurons = num_neurons
    self.num_k = num_k
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
    self.a_shape = [self.num_neurons]

  def step_inference(self, a_in, step):
    with tf.variable_scope("update_a"+str(step)) as scope:
      recon = self.build_decoder(a_in)
      error = self.data_tensor - recon
      ff_act = tf.matmul(error, self.w, name="driving_input")
      #Find maximum value and index
      max_idx = tf.argmax(ff_act, axis=-1)
      max_act = tf.reduce_max(ff_act, axis=-1)
      new_a = tf.one_hot(max_idx, depth=self.num_neurons) * max_act[..., tf.newaxis]
      a_out = a_in + new_a
    return a_out

  def infer_coefficients(self):
   a_list = [self.a_zeros]
   for step in range(self.num_k):
     a = self.step_inference(a_list[step], step)
     a_list.append(a)
   return a_list

  def build_decoder(self, input_tensor, name=None):
    return tf.matmul(input_tensor, tf.transpose(self.w), name=name)

  def compute_recon_loss(self, recon):
    with tf.variable_scope("unsupervised"):
      reduc_dim = list(range(1, len(recon.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.data_tensor, recon)),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def build_graph(self):
    with tf.variable_scope(self.variable_scope) as scope:
      with tf.variable_scope("constants") as scope:
        a_full_shape = tf.stack([tf.shape(self.data_tensor)[0]]+self.a_shape)
        self.a_zeros = tf.zeros(shape=a_full_shape, dtype=tf.float32, name="a_zeros")

      w_norm_dim = list(range(len(self.w_shape)-1)) # normalize across input dim(s)
      with tf.variable_scope("weights") as scope:
        self.weight_scope = tf.get_variable_scope()
        w_init = tf.nn.l2_normalize(tf.truncated_normal(self.w_shape, mean=0.0,
          stddev=0.5, dtype=tf.float32), axis=w_norm_dim, epsilon=self.eps, name="w_init")
        self.w = tf.get_variable(name="w", dtype=tf.float32, initializer=w_init,
          trainable=True)
        self.trainable_variables[self.w.name] = self.w

      with tf.variable_scope("norm_weights") as scope:
        self.norm_w = self.w.assign(tf.nn.l2_normalize(self.w, axis=w_norm_dim,
          epsilon=self.eps, name="row_l2_norm"))

      with tf.variable_scope("inference") as scope:
        self.inference_scope = tf.get_variable_scope()
        a_list = self.infer_coefficients()
        self.a = tf.identity(a_list[-1], name="activity")

      with tf.variable_scope("output") as scope:
        self.reconstruction_list = []
        for a in a_list:
          self.reconstruction_list.append(self.build_decoder(a))
        self.reconstruction = self.reconstruction_list[-1]

      with tf.variable_scope("loss") as scope:
        self.recon_loss_list = []
        self.total_loss_list = []
        for recon, a in zip(self.reconstruction_list, a_list):
          recon_loss = self.compute_recon_loss(recon)
          total_loss = recon_loss
          self.recon_loss_list.append(recon_loss)
          self.total_loss_list.append(total_loss)

        self.loss_dict = {"recon_loss":self.recon_loss_list[-1]}
        self.total_loss = self.total_loss_list[-1]



