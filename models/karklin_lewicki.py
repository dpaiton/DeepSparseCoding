import numpy as np
import logging
import tensorflow as tf
from models.base_model import Model 

class karklin_lewicki(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)
    self.build_graph()
    Model.setup_graph(self, self.graph)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    rectify_u      [bool] If set, rectify layer 1 activity
    rectify_v      [bool] If set, rectify layer 2 activity
    norm_a         [bool] If set, l2 normalize layer 1 activity
    norm_weights   [bool] If set, l2 normalize weights after updates
    batch_size     [int] Number of images in a training batch
    num_pixels     [int] Number of pixels
    num_u          [int] Number of layer 1 elements
    num_v          [int] Number of layer 2 elements
    num_steps      [int] Number of inference steps
    u_step_size    [int] Multiplier for gradient descent inference process
    v_step_size    [int] Multiplier for gradient descent inference process
  """
  def load_params(self, params):
    Model.load_params(self, params)
    # Meta parameters
    self.rectify_u = bool(params["rectify_u"])
    self.rectify_v = bool(params["rectify_v"])
    self.norm_a = bool(params["norm_a"])
    self.norm_weights = bool(params["norm_weights"])
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_u = int(params["num_u"])
    self.num_v = int(params["num_v"])
    self.a_shape = [self.num_pixels, self.num_u]
    self.b_shape = [self.num_u, self.num_v]
    # Hyper Parameters
    self.num_steps = int(params["num_steps"])
    self.u_step_size = int(params["u_step_size"])
    self.v_step_size = int(params["v_step_size"])

  """
  Build the nework described in:
    Y Karklin, MS Lewicki (2005) - A Hierarchical Bayesian Model for Learning
      Nonlinear Statistical Regularities in Nonstationary Natural Signals
  """
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32,
            shape=[self.num_pixels, None], name="input_data")
          self.recon_mult = tf.placeholder(tf.float32,
            shape=(), name="recon_mult")
          self.sparse_mult = tf.placeholder(tf.float32,
            shape=(), name="sparse_mult")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(
            shape=tf.pack([self.num_u, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="u_zeros")
          self.v_zeros = tf.zeros(
            shape=tf.pack([self.num_v, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="v_zeros")
          self.u_t_zeros = tf.zeros(
            shape=tf.pack([self.num_steps, self.num_u, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="u_t_zeros")
          self.v_t_zeros = tf.zeros(
            shape=tf.pack([self.num_steps, self.num_v, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="v_t_zeros")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        with tf.variable_scope("weights") as scope:
          self.a = tf.get_variable(name="a", dtype=tf.float32,
            initializer=tf.truncated_normal(self.a_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="a_init"), trainable=True)
          self.b = tf.get_variable(name="b", dtype=tf.float32,
            initializer=tf.truncated_normal(self.b_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="b_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_a = self.a.assign(tf.nn.l2_normalize(self.a,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.norm_b = self.b.assign(tf.nn.l2_normalize(self.b,
            dim=1, epsilon=self.eps, name="col_l2_norm"))
          self.normalize_weights = tf.group(self.norm_a, self.norm_b,
            name="do_normalization")

        with tf.name_scope("layers") as scope:
          self.u = tf.Variable(self.u_zeros, trainable=False,
            validate_shape=False, name="u")
          self.v = tf.Variable(self.v_zeros, trainable=False,
            validate_shape=False, name="v")
          self.clear_u = tf.group(self.u.assign(self.u_zeros))
          self.clear_v = tf.group(self.v.assign(self.v_zeros))
          self.sigma = tf.exp(-tf.matmul(self.b, self.v, name="sigma"))

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.a, self.u, name="reconstruction")

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.recon_loss = self.recon_mult * tf.reduce_mean(0.5 *
              tf.reduce_sum(tf.pow(tf.sub(self.x, self.x_), 2.0),
              reduction_indices=[0]))
            self.sparse_loss = self.sparse_mult * tf.reduce_mean(
              tf.reduce_sum(tf.abs(self.v), reduction_indices=[0]))
            self.feedback_loss = tf.reduce_mean(tf.reduce_sum(tf.add(
              tf.div(self.u, self.sigma), tf.log(self.sigma)),
              reduction_indices=[0]))
          self.total_loss = (self.recon_loss + self.feedback_loss +
            self.sparse_loss)

        with tf.name_scope("inference") as scope:
          self.u_t = [tf.Variable(self.u_zeros, trainable=False,
            validate_shape=False, name="u_t")
            for _ in np.arange(self.num_steps)]
          self.v_t = [tf.Variable(self.v_zeros, trainable=False,
            validate_shape=False, name="v_t")
            for _ in np.arange(self.num_steps)]
          self.du = -tf.gradients(self.total_loss, self.u)[0]
          self.dv = -tf.gradients(self.total_loss, self.v)[0]
          for step in np.arange(1, self.num_steps, dtype=np.int32):
            self.u_t[step] = (self.u_t[step-1]
              + self.u_step_size * self.du)
            self.v_t[step] = (self.v_t[step-1]
              + self.v_step_size * self.dv)
          if self.rectify_u:
            self.u = tf.abs(self.u_t[-1])
          else:
            self.u = self.u_t[-1]
          if self.rectify_v:
            self.v = tf.abs(self.v_t[-1])
          else:
            self.v = self.v_t[-1]

    self.graph_built = True

  """
  Log train progress information
  Inputs:
    input_data: load_MNIST data object containing the current image batch
    input_label: load_MNIST data object containing the current label batch
    batch_step: current batch number within the schedule
  """
  def print_update(self, input_data, input_label=None, batch_step=0):
    current_step = self.global_step.eval()
    feed_dict = self.get_feed_dict(input_data, input_label)
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    v_vals = tf.get_default_session().run(self.v, feed_dict)
    logging.info("Global batch index is %g"%(current_step))
    logging.info("Finished step %g out of %g for schedule %g"%(batch_step,
      self.get_sched("num_batches"), self.sched_idx))
    logging.info("\tloss:\t%g"%(
      self.total_loss.eval(feed_dict)))
    logging.info("\tmax val of u:\t\t%g"%(u_vals.max()))
    logging.info("\tmax val of v:\t\t%g"%(v_vals.max()))
    logging.info("\tl1 percent active:\t\t%0.2f%%"%(
      100.0 * np.count_nonzero(u_vals)
      / float(self.num_u * self.batch_size)))
    logging.info("\tl2 percent active:\t\t%0.2f%%"%(
      100.0 * np.count_nonzero(v_vals)
      / float(self.num_v * self.batch_size)))

  """
  Plot weights, reconstruction, and gradients
  Inputs: input_data and input_label used for the session
  """
  def generate_plots(self, input_image, input_label):
    feed_dict = self.get_feed_dict(input_image, input_label)
    current_step = str(self.global_step.eval())
    pf.save_data_tiled(
      self.w.eval().reshape(self.num_classes,
      int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
      normalize=True, title="Classification matrix at step number "
      +current_step, save_filename=(self.disp_dir+"w_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    pf.save_data_tiled(
      tf.transpose(self.phi).eval().reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step,
      save_filename=(self.disp_dir+"phi_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      if name == "phi":
        pf.save_data_tiled(grad.T.reshape(self.num_neurons,
          int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
          normalize=True, title="Gradient for phi at step "+current_step,
          save_filename=(self.disp_dir+"dphi_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
      elif name == "w":
        pf.save_data_tiled(grad.reshape(self.num_classes,
          int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
          normalize=True, title="Gradient for w at step "+current_step,
          save_filename=(self.disp_dir+"dw_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
