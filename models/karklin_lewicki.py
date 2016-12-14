import numpy as np
import logging
import tensorflow as tf
import utils.plot_functions as pf
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
  """
  def load_params(self, params):
  #TODO: Add rectify_u and rectify_v
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

  """
  Returns total loss function for given input
  Outputs:
    total_loss [float32] loss from Karklin & Lewicki
  Inputs:
    input_data [] 
    u_state []
    v_state []
    a []
    b []
  """
  def compute_loss(self, input_data, u_state=None, v_state=None, a=None,
    b=None):
    with tf.variable_scope("layers", reuse=True) as scope:
      if u_state is None:
        u_state = tf.get_variable("u")
      if v_state is None:
        v_state = tf.get_variable("v")
    with tf.variable_scope("weights", reuse=True) as scope:
      if a is None:
        a = tf.get_variable(name="a")
      if b is None:
        b = tf.get_variable(name="b")
    x_ = tf.matmul(a, u_state)
    sigma = tf.exp(-tf.matmul(b, v_state))
    recon_loss = self.recon_mult * tf.reduce_mean(0.5 *
      tf.reduce_sum(tf.pow(tf.sub(input_data, x_), 2.0),
      reduction_indices=[0]))
    feedback_loss = tf.reduce_mean(tf.reduce_sum(tf.add(
      tf.abs(tf.div(u_state, sigma)), tf.log(sigma)),
      reduction_indices=[0]))
    sparse_loss = self.sparse_mult * tf.reduce_mean(
      tf.reduce_sum(tf.abs(v_state), reduction_indices=[0]))
    total_loss = (recon_loss + feedback_loss + sparse_loss)
    return (total_loss, recon_loss, feedback_loss, sparse_loss)

  """
  Build the nework described in:
    Y Karklin, MS Lewicki (2005) - A Hierarchical Bayesian Model for Learning
      Nonlinear Statistical Regularities in Nonstationary Natural Signals
  Method for unrolling inference into the graph
    # compute loss for current sate
    # take gradient of loss wrt current state
    # compute new state = current state + eta * gradient
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
          self.u_step_size = tf.placeholder(tf.float32,
            shape=(), name="u_step_size")
          self.v_step_size = tf.placeholder(tf.float32,
            shape=(), name="v_step_size")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(
            shape=tf.pack([self.num_u, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="u_zeros")
          self.v_zeros = tf.zeros(
            shape=tf.pack([self.num_v, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="v_zeros")

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
        
        with tf.variable_scope("layers") as scope:
          self.u = tf.get_variable(name="u", dtype=tf.float32,
            initializer=self.u_zeros, trainable=False, validate_shape=False)
          self.v = tf.get_variable(name="v", dtype=tf.float32,
            initializer=self.v_zeros, trainable=False, validate_shape=False)

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.a, self.u, name="reconstruction")
          with tf.name_scope("l1_prior"):
            self.sigma = tf.exp(-tf.matmul(self.b, self.v, name="sigma"))

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            (self.total_loss,
            self.recon_loss,
            self.feedback_loss,
            self.sparse_loss) = self.compute_loss(self.x)

        with tf.name_scope("inference") as scope:
          self.clear_u = tf.group(self.u.assign(self.u_zeros))
          self.clear_v = tf.group(self.v.assign(self.v_zeros))
          current_loss = []
          self.u_t = [self.u_zeros]
          self.v_t = [self.v_zeros]
          for step in range(self.num_steps):
            current_loss.append(self.compute_loss(self.x, self.u_t[step],
              self.v_t[step])[0])
            du = -tf.gradients(current_loss[step], self.u_t[step])[0]
            dv = -tf.gradients(current_loss[step], self.v_t[step])[0]
            self.u_t.append(self.u_t[step] + self.u_step_size * du)
            self.v_t.append(self.v_t[step] + self.v_step_size * dv)
          self.do_inference = tf.group(self.u.assign(self.u_t[-1]),
            self.v.assign(self.v_t[-1]), name="do_inference")

    self.graph_built = True

  """
  Log train progress information
  Inputs:
    input_data: data object containing the current image batch
    input_label: data object containing the current label batch
    batch_step: current batch number within the schedule
  """
  def print_update(self, input_data, input_label=None, batch_step=0):
  # TODO: Why did I have to get defult session for u/v and not loss?
    current_step = self.global_step.eval()
    feed_dict = self.get_feed_dict(input_data, input_label)
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    v_vals = tf.get_default_session().run(self.v, feed_dict)
    logging.info("Global batch index is %g"%(current_step))
    logging.info("Finished step %g out of %g for schedule %g"%(batch_step,
      self.get_sched("num_batches"), self.sched_idx))
    logging.info("\trecon loss:\t\t%g"%(
      self.recon_loss.eval(feed_dict)))
    logging.info("\tfeedback loss:\t\t%g"%(
      self.feedback_loss.eval(feed_dict)))
    logging.info("\tsparse loss:\t\t%g"%(
      self.sparse_loss.eval(feed_dict)))
    logging.info("\ttotal loss:\t\t%g"%(
      self.total_loss.eval(feed_dict)))
    logging.info("\tmax val of u:\t\t%g"%(u_vals.max()))
    logging.info("\tmax val of v:\t\t%g"%(v_vals.max()))
    logging.info("\tl1 percent active:\t%0.2f%%"%(
      100.0 * np.count_nonzero(u_vals)
      / float(self.num_u * self.batch_size)))
    logging.info("\tl2 percent active:\t%0.2f%%"%(
      100.0 * np.count_nonzero(v_vals)
      / float(self.num_v * self.batch_size)))

  """
  Plot weights, reconstruction, and gradients
  Inputs: input_data and input_label used for the session
  """
  def generate_plots(self, input_data, input_label=None):
    feed_dict = self.get_feed_dict(input_data, input_label)
    current_step = str(self.global_step.eval())
    pf.save_data_tiled(
      tf.transpose(self.b).eval().reshape(self.num_v,
      int(np.sqrt(self.num_u)), int(np.sqrt(self.num_u))),
      normalize=True, title="Density weights matrix at step number "
      +current_step, save_filename=(self.disp_dir+"b_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    pf.save_data_tiled(
      tf.transpose(self.a).eval().reshape(self.num_u,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step,
      save_filename=(self.disp_dir+"a_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      if name == "a":
        pf.save_data_tiled(grad.T.reshape(self.num_u,
          int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
          normalize=True, title="Gradient for a at step "+current_step,
          save_filename=(self.disp_dir+"da_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
      elif name == "b":
        pf.save_data_tiled(grad.reshape(self.num_v,
          int(np.sqrt(self.num_u)), int(np.sqrt(self.num_u))),
          normalize=True, title="Gradient for b at step "+current_step,
          save_filename=(self.disp_dir+"db_v"+self.version+"_"
          +current_step.zfill(5)+".pdf"))
