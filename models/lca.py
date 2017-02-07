import numpy as np
import logging
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class LCA(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)
    self.build_graph()
    Model.setup_graph(self, self.graph)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    rectify_a    [bool] If set, rectify layer 1 activity
    norm_weights [bool] If set, l2 normalize weights after updates
    batch_size   [int] Number of images in a training batch
    num_pixels   [int] Number of pixels
    num_neurons  [int] Number of LCA neurons
    num_steps    [int] Number of inference steps
    dt           [float] Discrete global time constant
    tau          [float] LCA time constant
    thresh_type  [str] "hard" or "soft" - LCA threshold function specification
  """
  def load_params(self, params):
    Model.load_params(self, params)
    # Meta parameters
    self.rectify_a = bool(params["rectify_a"])
    self.norm_weights = bool(params["norm_weights"])
    self.thresh_type = str(params["thresh_type"])
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_neurons = int(params["num_neurons"])
    self.phi_shape = [self.num_pixels, self.num_neurons]
    # Hyper Parameters
    self.num_steps = int(params["num_steps"])
    self.dt = float(params["dt"])
    self.tau = float(params["tau"])
    self.eta = self.dt / self.tau

  """
  Performs threshold function on input values
  Outputs:
    a [float32] thresholded neuron output
  Inputs:
    u []
    thresh_type [str] "hard" or "soft"
  """
  def compute_output(self):
    return a

  """Build the TensorFlow graph object"""
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(
            tf.float32, shape=[self.num_pixels, None], name="input_data")
          self.sparse_mult = tf.placeholder(
            tf.float32, shape=(), name="sparse_mult")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(
            shape=tf.pack([self.num_neurons, tf.shape(self.x)[1]]),
            dtype=tf.float32, name="u_zeros")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="phi_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_phi,
            name="l2_normalization")

        with tf.name_scope("inference") as scope:
          self.u = tf.Variable(self.u_zeros, trainable=False,
            validate_shape=False, name="u")
          if self.thresh_type == "soft":
            if self.rectify_a:
              self.a = tf.select(tf.greater(self.u, self.sparse_mult),
                tf.sub(self.u, self.sparse_mult), self.u_zeros, name="activity")
            else:
              self.a = tf.select(tf.greater(self.u, self.sparse_mult),
                tf.sub(self.u, self.sparse_mult), tf.select(tf.less(self.u,
                -self.sparse_mult), tf.add(self.u, self.sparse_mult),
                self.u_zeros), name="activity")
          elif self.thresh_type == "hard":
            if self.rectify_a:
              self.a = tf.select(tf.greater(self.u, self.sparse_mult), self.u,
                self.u_zeros, name="activity")
            else:
              self.a = tf.select(tf.greater(self.u, self.sparse_mult),
                self.u, tf.select(tf.less(self.u, -self.sparse_mult), self.u,
                self.u_zeros), name="activity")

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.phi, self.a, name="reconstruction")

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.recon_loss = tf.reduce_mean(0.5 *
              tf.reduce_sum(tf.pow(tf.sub(self.x, self.x_), 2.0),
              reduction_indices=[0]), name="recon_loss")
            self.sparse_loss = self.sparse_mult * tf.reduce_mean(
              tf.reduce_sum(tf.abs(self.a), reduction_indices=[0]),
              name="sparse_loss")
            self.unsupervised_loss = (self.recon_loss + self.sparse_loss)
          self.total_loss = self.unsupervised_loss

        with tf.name_scope("update_u") as scope:
          self.lca_b = tf.matmul(tf.transpose(self.phi), self.x,
            name="driving_input")
          self.lca_g = (tf.matmul(tf.transpose(self.phi), self.phi,
            name="gram_matrix") -
            tf.constant(np.identity(int(self.phi_shape[1]), dtype=np.float32),
            name="identity_matrix"))
          self.lca_explain_away = tf.matmul(self.lca_g, self.a,
            name="explaining_away")
          self.du = self.lca_b - self.lca_explain_away - self.u
          self.step_inference = tf.group(self.u.assign_add(self.eta * self.du),
            name="step_inference")
          self.clear_activity = tf.group(self.u.assign(self.u_zeros),
            name="clear_activity")

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.pow(tf.sub(self.x, self.x_), 2.0),
              name="mean_squared_error")
            self.pSNRdB = tf.mul(10.0, tf.log(tf.div(tf.pow(1.0, 2.0), MSE)),
              name="recon_quality")
    self.graph_built = True

  """
  Log train progress information
  Inputs:
    input_data: data object containing the current image batch
    input_labels: data object containing the current label batch
    batch_step: current batch number within the schedule
  NOTE: Casting tf.eval output to an np.array and then to a list is required to
    ensure that the data type is valid for js.dumps(). An alternative would be
    to write an np function that converts numpy types to their corresponding
    python types.
  """
  def print_update(self, input_data, input_labels=None, batch_step=0):
    # TODO: When is it required to get defult session?
    Model.print_update(self, input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval()).tolist()
    recon_loss = np.array(self.recon_loss.eval(feed_dict)).tolist()
    sparse_loss = np.array(self.sparse_loss.eval(feed_dict)).tolist()
    total_loss = np.array(self.total_loss.eval(feed_dict)).tolist()
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max()).tolist()
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.num_neurons * self.batch_size)).tolist()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_sched("num_batches"),
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act}
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      stat_dict[name+"_max_grad"] = np.array(grad.max()).tolist()
      stat_dict[name+"_min_grad"] = np.array(grad.min()).tolist()
    js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
    self.log_info("<stats>"+js_str+"</stats>")

  """
  Plot weights, reconstruction, and gradients
  Inputs:
    input_data: data object containing the current image batch
    input_labels: data object containing the current label batch
  """
  def generate_plots(self, input_data, input_labels=None):
    Model.generate_plots(self, input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = str(self.global_step.eval())
    pf.save_data_tiled(
      tf.transpose(self.phi).eval().reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"
      +current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      pf.save_data_tiled(grad.T.reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Gradient for phi at step "+current_step,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"
        +current_step.zfill(5)+".pdf"))
