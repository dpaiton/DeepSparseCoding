import numpy as np
import logging
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

import os

class density_learner(Model):
  def __init__(self, params, schedule):
    super(density_learner, self).__init__(params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      rectify_a    [bool] If set, rectify layer 1 activity
      norm_weights [bool] If set, rescale the density weights using an l2 norm
      batch_size   [int] Number of images in a training batch
      num_pixels   [int] Number of pixels
      num_neurons  [int] Number of LCA neurons
      num_u_steps  [int] Number of u inference steps
      num_v_steps  [int] Number of v inference steps
      dt           [float] Discrete global time constant
      tau          [float] LCA time constant
    """
    super(density_learner, self).load_params(params)
    # Meta parameters
    self.rectify_a = bool(params["rectify_a"])
    self.rectify_v = bool(params["rectify_v"])
    self.norm_weights = bool(params["norm_weights"])
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_neurons = int(params["num_neurons"])
    self.num_v = int(params["num_v"])
    self.phi_shape = [self.num_pixels, self.num_neurons]
    self.b_shape = [self.num_neurons, self.num_v]
    # Hyper Parameters
    self.num_u_steps = int(params["num_u_steps"])
    self.num_v_steps = int(params["num_v_steps"])
    self.v_step_scale = float(params["v_step_scale"])
    self.dt = float(params["dt"])
    self.tau = float(params["tau"])
    self.eta = self.dt / self.tau
    self.v_step_size = self.v_step_scale# * self.eta


  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=[None, self.num_pixels],
            name="input_data")
          self.u_fb_mult = tf.placeholder(tf.float32, shape=(),
            name="u_fb_mult")
          self.v_sparse_mult = tf.placeholder(tf.float32, shape=(),
            name="v_sparse_mult")
          self.b_decay_mult = tf.placeholder(tf.float32, shape=(),
            name="b_decay_mult")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(
            shape=tf.stack([tf.shape(self.x)[0], self.num_neurons]),
            dtype=tf.float32, name="u_zeros")
          self.small_v = tf.multiply(0.1, tf.ones(
            shape=tf.stack([tf.shape(self.x)[0], self.num_v]),
            dtype=tf.float32, name="v_ones"), name="small_v_init")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          phi_init = tf.nn.l2_normalize(np.load((os.path.expanduser("~")
            +"/Work/Projects/pretrain_white/analysis/0.0/weights/phi.npz"))["data"],
            dim=0, epsilon=self.eps, name="phi_init")
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=phi_init, trainable=False)
          #b_init = tf.multiply(1e-1, tf.ones(self.b_shape,
          #  dtype=tf.float32), name="b_init")
          #b_init = tf.tile(tf.truncated_normal([1, self.b_shape[1]], mean=0.0,
          #  stddev=1.0, dtype=tf.float32), [self.b_shape[0], 1], name="b_init")
          b_init = tf.truncated_normal(self.b_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="b_init")
          self.b = tf.get_variable(name="b", dtype=tf.float32,
            initializer=b_init, trainable=True)

        with tf.name_scope("norm_weights") as scope:
          self.l2_norm_b = self.b.assign(tf.nn.l2_normalize(self.b,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.norm_weights = tf.group(self.l2_norm_b,
            name="l2_normalization")

        with tf.name_scope("inference") as scope:
          self.u = tf.Variable(self.u_zeros, trainable=False,
            validate_shape=False, name="u")
          self.v = tf.Variable(self.small_v, trainable=False,
            validate_shape=False, name="v")
          self.sigma = tf.exp(tf.matmul(self.v, tf.transpose(self.b)),
            name="sigma")
          #self.fb_sparse_mult = tf.div(self.u_fb_mult, self.sigma)
          if self.rectify_a:
            self.a = tf.where(tf.greater(self.u, self.u_fb_mult),
              tf.subtract(self.u, self.u_fb_mult), self.u_zeros,
              name="activity")
          else:
            self.a = tf.where(tf.greater(self.u, self.u_fb_mult),
              tf.subtract(self.u, self.u_fb_mult),
              tf.where(tf.less(self.u, -self.u_fb_mult),
              tf.add(self.u, self.u_fb_mult), self.u_zeros),
              name="activity")

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.a, tf.transpose(self.phi),
              name="reconstruction")
          with tf.name_scope("layer_0_estimate"):
            self.a_recon = tf.divide(self.a, self.sigma)

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.recon_loss = tf.reduce_mean(tf.multiply(0.5,
              tf.reduce_sum(tf.square(tf.subtract(self.x, self.x_)),
              axis=[1])), name="recon_loss")
            self.fb_loss = tf.reduce_mean(tf.multiply(self.u_fb_mult,
              tf.reduce_sum(tf.add(self.a_recon, tf.log(self.sigma)),
              axis=[1])), name="fb_loss")
            self.density_loss = tf.reduce_mean(tf.multiply(self.v_sparse_mult,
              tf.reduce_sum(tf.abs(self.v), axis=[1])), name="density_loss")
            self.b_loss = tf.reduce_mean(tf.multiply(self.b_decay_mult,
              tf.reduce_sum(tf.abs(self.b), axis=[1])), name="b_loss")
            self.unsupervised_loss = tf.add_n([self.recon_loss,
              self.fb_loss, self.density_loss, self.b_loss],
              name="unsupervised_loss")
          self.total_loss = self.unsupervised_loss

        with tf.name_scope("update_u") as scope:
          self.lca_b = tf.matmul(self.x, self.phi, name="driving_input")
          self.lca_g = (tf.matmul(tf.transpose(self.phi), self.phi,
            name="gram_matrix") -
            tf.constant(np.identity(self.phi_shape[1], dtype=np.float32),
            name="identity_matrix"))
          self.lca_explain_away = tf.matmul(self.a, self.lca_g,
            name="explaining_away")
          self.du = self.lca_b - self.lca_explain_away - self.u
          self.step_u = tf.group(self.u.assign_add(self.eta * self.du),
            name="step_u")

        with tf.name_scope("update_v") as scope:
          with tf.control_dependencies([self.step_u]):
            #self.dv = -tf.gradients(self.total_loss, self.v)[0]
            a_recon_err = tf.multiply(self.u_fb_mult, tf.subtract(1.0,
              self.a_recon))
            projected_err = tf.matmul(a_recon_err, self.b)
            dedv = tf.add(projected_err, tf.multiply(self.v_sparse_mult,
              tf.sign(self.v)), name="dv")
            self.dv = -self.v_step_size * dedv
            if self.rectify_v:
              self.step_v = self.v.assign(tf.nn.relu(tf.add(self.v, self.dv),
              name="new_v"))
            else:
              self.step_v = self.v.assign_add(self.dv, name="step_v")

        self.step_inference = tf.group(self.step_u, self.step_v,
          name="step_inference")
        self.reset_activity = tf.group(self.u.assign(self.u_zeros),
          self.v.assign(self.small_v), name="reset_activity")

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.pow(tf.subtract(self.x, self.x_), 2.0),
              axis=[1, 0], name="mean_squared_error")
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.pow(1.0,
               2.0), MSE)), name="recon_quality")
    self.graph_built = True

  #def compute_weight_gradients(self, optimizer, weight_op=None):
  #  """
  #  Returns the manually-computed gradient
  #  NOTE:
  #    This child function does not use optimizer input
  #    weight_op must be a list with a single matrix ("self.b") in it
  #  """
  #  assert len(weight_op) == 1, ("density_learner should only have one"
  #    +"weight matrix")
  #  weight_name = weight_op[0].name.split('/')[1].split(':')[0]#np.split
  #  op1 = tf.subtract(1.0, self.a_recon)
  #  op2 = tf.transpose(op1)
  #  op3 = tf.matmul(op2, self.v)
  #  gradient = tf.multiply(self.u_fb_mult, tf.divide(op3,
  #    2.0*float(self.batch_size)))
  #  #gradient = -tf.divide(tf.multiply(0.5/float(self.batch_size),
  #  #  tf.matmul(tf.transpose(tf.subtract(tf.divide(tf.abs(self.a),
  #  #  self.sigma), 1.0)), self.v)), name=weight_name+"_gradient")
  #  return [(gradient, weight_op[0])]

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    NOTE: Casting tf.eval output to an np.array and then to a list is required to
      ensure that the data type is valid for js.dumps(). An alternative would be
      to write a numpy function that converts numpy types to their corresponding
      python types.
    """
    super(density_learner, self).print_update(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval())
    recon_loss = np.array(self.recon_loss.eval(feed_dict))
    fb_loss = np.array(self.fb_loss.eval(feed_dict))
    density_loss = np.array(self.density_loss.eval(feed_dict))
    b_loss = np.array(self.b_loss.eval(feed_dict))
    total_loss = np.array(self.total_loss.eval(feed_dict))
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.num_neurons * self.batch_size))
    v_vals = tf.get_default_session().run(self.v, feed_dict)
    v_vals_max = np.array(v_vals.max())
    v_frac_act = np.array(np.count_nonzero(v_vals)
      / float(self.num_v * self.batch_size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "fb_loss":fb_loss,
      "density_loss":density_loss,
      "b_loss":b_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act,
      "v_max":v_vals_max,
      "v_fraction_active":v_frac_act}
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      stat_dict[name+"_max_grad"] = np.array(grad.max())
      stat_dict[name+"_min_grad"] = np.array(grad.min())
    js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(density_learner, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = str(self.global_step.eval())
    #recon = tf.get_default_session().run(self.x_, feed_dict)
    #weights = tf.get_default_session().run(self.phi, feed_dict)
    b_weights = tf.get_default_session().run(self.b, feed_dict)
    v_vals = tf.get_default_session().run(self.v, feed_dict)
    #pf.plot_data_tiled(input_data.reshape((self.batch_size,
    #  np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
    #  normalize=False, title="Images at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"images_"+self.version+"-"+current_step.zfill(5)+".pdf"))
    #pf.plot_data_tiled(weights.T.reshape(self.num_neurons,
    #  int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
    #  normalize=False, title="Dictionary at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"phi_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
    pf.plot_data_tiled(b_weights.T.reshape(self.num_v,
      int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
      normalize=True, title="Density weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"b_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
    #pf.plot_bar(np.linalg.norm(weights, axis=1, keepdims=False), num_xticks=5,
    #  title="phi l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
    #  save_filename=(self.disp_dir+"phi_norm_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    pf.plot_bar(np.linalg.norm(b_weights, axis=1, ord=1, keepdims=False),
      num_xticks=5, title="b l1 norm", xlabel="Basis Index", ylabel="L1 Norm",
      save_filename=(self.disp_dir+"b_norm_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    pf.plot_activity_hist(v_vals, num_bins=1000,
      title="v Activity Histogram at step "+current_step,
      save_filename=(self.disp_dir+"v_hist_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    #pf.plot_data_tiled(recon.reshape((self.batch_size,
    #  np.int(np.sqrt(self.num_pixels)), np.int(np.sqrt(self.num_pixels)))),
    #  normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      if name == "phi":
        pf.plot_data_tiled(grad.T.reshape(self.num_neurons,
          int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
          normalize=True, title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
      elif name == "b":
        pf.plot_data_tiled(grad.T.reshape(self.num_v,
          int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
          normalize=True, title="Gradient for b at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"db_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
