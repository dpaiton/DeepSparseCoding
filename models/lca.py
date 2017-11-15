import numpy as np
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class LCA(Model):
  def __init__(self, params, schedule):
    super(LCA, self).__init__(params, schedule)

  def load_params(self, params):
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
    super(LCA, self).load_params(params)
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

  def compute_excitatory_current(self):
    return tf.matmul(self.x, self.phi, name="driving_input")

  def compute_inhibitory_connectivity(self):
   return (tf.matmul(tf.transpose(self.phi), self.phi, name="gram_matrix")
     - tf.constant(np.identity(self.phi_shape[1], dtype=np.float32), name="identity_matrix"))

  def step_inference(self, u_in, a_in, b, g, step):
    with tf.name_scope("update_u"+str(step)) as scope:
      lca_explain_away = tf.matmul(a_in, g, name="explaining_away")
      du = tf.subtract(tf.subtract(b, lca_explain_away), u_in, name="du")
      u_out = tf.add(u_in, tf.multiply(self.eta, du))
    return u_out, lca_explain_away

  def threshold_units(self, u_in):
    if self.thresh_type == "soft":
      if self.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult), self.u_zeros)
      else:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult),
          tf.subtract(u_in, self.sparse_mult),
          tf.where(tf.less(u_in, -self.sparse_mult),
          tf.add(u_in, self.sparse_mult),
          self.u_zeros))
    elif self.thresh_type == "hard":
      if self.rectify_a:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in,
          self.u_zeros)
      else:
        a_out = tf.where(tf.greater(u_in, self.sparse_mult), u_in,
          tf.where(tf.less(u_in, -self.sparse_mult), u_in, self.u_zeros))
    return a_out

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
    return tf.matmul(a_in, tf.transpose(self.phi), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.x, self.compute_recon(a_in))),
        axis=[1]), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      sparse_loss = self.sparse_mult * tf.reduce_mean(
        tf.reduce_sum(tf.abs(a_in), axis=[1]), name="sparse_loss")
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
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=[None, self.num_pixels], name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(shape=tf.stack([tf.shape(self.x)[0], self.num_neurons]),
            dtype=tf.float32, name="u_zeros")
          self.u_noise = tf.truncated_normal(
            shape=tf.stack([tf.shape(self.x)[0], self.num_neurons]),
            mean=0.0, stddev=0.1, dtype=tf.float32, name="u_noise")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          phi_init = tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=0.5, dtype=tf.float32, name="phi_init")
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=phi_init, trainable=True)

        with tf.name_scope("norm_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.norm_weights = tf.group(self.norm_phi,
            name="l2_normalization")

        with tf.variable_scope("inference") as scope:
         u_list, a_list = self.infer_coefficients()
         self.u = tf.identity(u_list[-1], name="u")
         self.a = tf.identity(a_list[-1], name="activity")

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = self.compute_recon(self.a)

        with tf.name_scope("loss") as scope:
          loss_funcs = self.get_loss_funcs()
          self.loss_dict = dict(zip(
            [key for key in loss_funcs.keys()], [func(self.a) for func in loss_funcs.values()]))
          self.total_loss = self.compute_total_loss(self.a, loss_funcs)

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.x_)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    super(LCA, self).print_update(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.total_loss, self.a]
    grad_name_list = []
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      eval_list.append(weight_grad_var[0][0])
      grad_name_list.append(weight_grad_var[0][1].name.split('/')[1].split(':')[0])#2nd is np.split
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, a_vals = out_vals[0:5]
    grads = out_vals[5:]
    a_vals_max = np.array(a_vals.max())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.batch_size * self.num_neurons))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act}
    for grad, name in zip(grads, grad_name_list):
      stat_dict[name+"_max_grad"] = np.array(grad.max())
      stat_dict[name+"_min_grad"] = np.array(grad.min())
    js_str = self.js_dumpstring(stat_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(LCA, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = str(self.global_step.eval())
    recon = tf.get_default_session().run(self.x_, feed_dict)
    weights = tf.get_default_session().run(self.phi, feed_dict)
    pf.plot_data_tiled(input_data.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    pf.plot_data_tiled(weights.T.reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=False, title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_bar(np.linalg.norm(weights, axis=1, keepdims=False), num_xticks=5,
      title="phi l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"phi_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_data_tiled(recon.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(grad.T.reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".png"))
