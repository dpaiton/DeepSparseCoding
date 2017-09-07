import numpy as np
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

import os

class dsc(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    rectify_v      [bool] If set, rectify layer 2 activity
    norm_weights   [bool] If set, l2 normalize weights after updates
    batch_size     [int] Number of images in a training batch
    num_pixels     [int] Number of pixels
    num_u          [int] Number of layer 1 elements
    num_v          [int] Number of layer 2 elements
    num_steps      [int] Number of inference steps
  """
  def load_params(self, params):
    Model.load_params(self, params)
    # Meta parameters
    self.rectify_u = bool(params["rectify_u"])
    self.rectify_v = bool(params["rectify_v"])
    self.w_init_loc = params["w_init_loc"]
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_u = int(params["num_u"])
    self.num_v = int(params["num_v"])
    self.a_shape = [self.num_pixels, self.num_u]
    self.b_shape = [self.num_u, self.num_v]
    self.w_shapes = [vals for vals in zip(self.a_shape, self.b_shape)]
    # Hyper Parameters
    self.num_steps = int(params["num_steps"])

  """Check parameters with assertions"""
  def check_params(self):
    Model.check_params(self)
    assert np.sqrt(self.num_u) == np.floor(np.sqrt(self.num_u)), (
      "The parameter `num_u` must have an even square-root for plotting.")

  """
  Returns total loss function for given input
  Outputs:
    total_loss [float32] loss adapted from Karklin & Lewicki
  Inputs:
    input_data []
    u_state []
    v_state []
  """
  def compute_loss(self, input_data, u_state, v_state):
    with tf.variable_scope("weights", reuse=True) as scope:
      a_state = tf.get_variable(name="a")
      b_state = tf.get_variable(name="b")
    with tf.name_scope("iterative_loss"):
      temp_sigma = tf.exp(tf.matmul(v_state, tf.transpose(b_state)))#0.83
      temp_x_ = tf.matmul(u_state, tf.transpose(a_state))
      self.input_stddev = 1.0#tf.reduce_mean(tf.square(tf.nn.moments(input_data,
        #axes=[1])[1]))
      recon_loss = tf.multiply(self.recon_mult, tf.reduce_mean(
        tf.multiply(tf.divide(1.0, (2.0*self.input_stddev)),
        tf.reduce_sum(tf.square(tf.subtract(input_data, temp_x_)),
        axis=[1]))), name="recon_loss")
      feedback_loss = tf.reduce_mean(tf.reduce_sum(tf.add(
        tf.divide(u_state, temp_sigma), tf.log(temp_sigma)),
        axis=[1]), name="fb_loss")
      sparse_loss = tf.multiply(self.sparse_mult, tf.reduce_mean(
        tf.reduce_sum(tf.abs(v_state), axis=[1])), name="sparse_loss")
      a_loss = tf.multiply(self.a_decay_mult, tf.reduce_mean(
        tf.reduce_sum(tf.square(a_state), axis=[1])), name="a_loss")
      b_loss = tf.multiply(self.b_decay_mult, tf.reduce_mean(
        tf.reduce_sum(tf.abs(b_state), axis=[1])), name="b_loss")
      total_loss = tf.add_n([recon_loss, feedback_loss, sparse_loss,
        a_loss, b_loss], name="total_loss")
    return (total_loss, recon_loss, feedback_loss, sparse_loss, a_loss, b_loss)

  def inference_step(self, inference_idx, u_in, v_in, u_relu, v_relu):
    current_u_loss = self.compute_loss(self.x, u_in, v_in)[0]
    with tf.name_scope("inference"+str(inference_idx)):
      u_grad = tf.gradients(current_u_loss, u_in)[0]
      u_update = tf.add(u_in, tf.multiply(-self.u_step_size, u_grad),
        name="u_update"+str(inference_idx))
      u_out = tf.nn.relu(u_update) if u_relu else u_update
    current_v_loss = self.compute_loss(self.x, u_out, v_in)[0]
    with tf.name_scope("inference"+str(inference_idx)):
      v_grad = tf.gradients(current_v_loss, v_in)[0]
      v_update = tf.add(v_in, tf.multiply(-self.v_step_size, v_grad),
        name="v_update"+str(inference_idx))
      v_out = tf.nn.relu(v_update) if v_relu else v_update
      return u_out, v_out

  """
  Based on the neworks described in:
    Y Karklin, MS Lewicki (2005) - A Hierarchical Bayesian Model for Learning
      Nonlinear Statistical Regularities in Nonstationary Natural Signals
    TS Lee, D Mumford (2003) - Hierarchical Bayesian Inference in the Visual
      Cortex
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
            shape=[None, self.num_pixels], name="input_data")
          self.recon_mult = tf.placeholder(tf.float32,
            shape=(), name="recon_mult")
          self.sparse_mult = tf.placeholder(tf.float32,
            shape=(), name="sparse_mult")
          self.a_decay_mult = tf.placeholder(tf.float32,
            shape=(), name="a_decay_mult")
          self.b_decay_mult = tf.placeholder(tf.float32,
            shape=(), name="b_decay_mult")
          self.u_step_size = tf.placeholder(tf.float32,
            shape=(), name="u_step_size")
          self.v_step_size = tf.placeholder(tf.float32,
            shape=(), name="v_step_size")

        with tf.name_scope("constants") as scope:
          self.u_noise = tf.truncated_normal(
            shape=tf.stack([tf.shape(self.x)[0], self.num_u]),
            mean=0.0, stddev=0.1, dtype=tf.float32, name="u_noise")
          self.v_ones = tf.ones(
            shape=tf.stack([tf.shape(self.x)[0], self.num_v]),
            dtype=tf.float32, name="v_ones")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        if self.w_init_loc is None:
          w_inits = [tf.truncated_normal(self.w_shapes[0], mean=0.0,
            stddev=0.5, dtype=tf.float32, name="a_init"),
            tf.multiply(0.1, tf.ones(self.w_shapes[1], dtype=tf.float32),
            name="b_init")]
        else:
          w_inits = [np.load(self.w_init_loc)["data"],
            tf.multiply(0.1, tf.ones(self.w_shapes[1], dtype=tf.float32),
            name="b_init")]

        with tf.variable_scope("weights") as scope:
          self.a = tf.get_variable(name="a", dtype=tf.float32,
            initializer=w_inits[0], trainable=True)
          self.b = tf.get_variable(name="b", dtype=tf.float32,
            initializer=w_inits[1], trainable=True)

        u_list = [self.u_noise]
        v_list = [self.v_ones]
        for inference_idx in range(self.num_steps-1):
          u_out, v_out = self.inference_step(inference_idx,
            u_list[inference_idx], v_list[inference_idx],
            self.rectify_u, self.rectify_v)
          u_list.append(u_out)
          v_list.append(v_out)

        with tf.variable_scope("layers") as scope:
          self.u = tf.get_variable(name="u", dtype=tf.float32,
            initializer=u_list[-1], trainable=False, validate_shape=False)
          self.v = tf.get_variable(name="v", dtype=tf.float32,
            initializer=v_list[-1], trainable=False, validate_shape=False)
        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.u, tf.transpose(self.a),
              name="reconstruction")
            MSE = tf.reduce_mean(tf.pow(tf.subtract(self.x, self.x_), 2.0),
              axis=[1, 0], name="mean_squared_error")
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(1.0),
              MSE)), name="recon_quality")
          with tf.name_scope("layer1_prior"):
            self.sigma = tf.exp(tf.matmul(self.v, tf.transpose(self.b),
              name="sigma"))

          with tf.name_scope("loss") as scope:
            with tf.name_scope("unsupervised"):
              (self.total_loss,
              self.recon_loss,
              self.feedback_loss,
              self.sparse_loss,
              self.a_loss,
              self.b_loss) = self.compute_loss(self.x, self.u, self.v)

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
    feedback_loss = np.array(self.feedback_loss.eval(feed_dict)).tolist()
    sparse_loss = np.array(self.sparse_loss.eval(feed_dict)).tolist()
    a_loss = np.array(self.a_loss.eval(feed_dict)).tolist()
    b_loss = np.array(self.b_loss.eval(feed_dict)).tolist()
    total_loss = np.array(self.total_loss.eval(feed_dict)).tolist()
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    u_vals_max = np.array(u_vals.max()).tolist()
    v_vals = tf.get_default_session().run(self.v, feed_dict)
    v_vals_max = np.array(v_vals.max()).tolist()
    u_frac_act = np.array(np.count_nonzero(u_vals)
      / float(self.num_u * self.batch_size)).tolist()
    v_frac_act = np.array(np.count_nonzero(v_vals)
      / float(self.num_v * self.batch_size)).tolist()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_sched("num_batches"),
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "feedback_loss":feedback_loss,
      "sparse_loss":sparse_loss,
      "a_l2_loss":a_loss,
      "b_l1_loss":b_loss,
      "total_loss":total_loss,
      "u_max":u_vals_max,
      "v_max":v_vals_max,
      "u_fraction_active":u_frac_act,
      "v_fraction_active":v_frac_act}
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      stat_dict[name+"_max_grad"] = np.array(grad.max()).tolist()
      stat_dict[name+"_min_grad"] = np.array(grad.min()).tolist()
    js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
    self.log_info("<stats>"+js_str+"</stats>")
    #print(self.input_stddev.eval(feed_dict))

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
    recon = tf.get_default_session().run(self.x_, feed_dict)
    a_weights = tf.get_default_session().run(self.a, feed_dict)
    #b_weights = tf.get_default_session().run(self.b, feed_dict)
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    #v_vals = tf.get_default_session().run(self.v, feed_dict)
    #pf.plot_data_tiled(input_data.reshape((self.batch_size,
    #  np.int(np.sqrt(self.num_pixels)),
    #  np.int(np.sqrt(self.num_pixels)))),
    #  normalize=False, title="Images at step "+current_step, vmin=np.min(input_data),
    #  vmax=np.max(input_data), save_filename=(self.disp_dir+"images_"+self.version+"-"
    #  +current_step.zfill(5)+".pdf"))
    pf.plot_data_tiled(recon.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    pf.plot_data_tiled(a_weights.T.reshape(self.num_u,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=False, title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"a_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    #pf.plot_data_tiled(b_weights.T.reshape(self.num_v,
    #  int(np.sqrt(self.num_u)), int(np.sqrt(self.num_u))),
    #  normalize=False, title="Density weights matrix at step number "+current_step,
    #  vmin=None, vmax=None, save_filename=(self.disp_dir+"b_v"+self.version+"-"
    #  +current_step.zfill(5)+".pdf"))
    pf.plot_activity_hist(u_vals, num_bins=1000,
      title="u Activity Histogram at step "+current_step,
      save_filename=(self.disp_dir+"u_hist_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    #pf.plot_activity_hist(v_vals, num_bins=1000,
    #  title="v Activity Histogram at step "+current_step,
    #  save_filename=(self.disp_dir+"v_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".pdf"))
    pf.plot_bar(np.linalg.norm(a_weights, axis=1, keepdims=False), num_xticks=5,
      title="a l2 norm", xlabel="Basis Index",ylabel="L2 Norm",
      save_filename=(self.disp_dir+"a_norm_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    #pf.plot_bar(np.linalg.norm(b_weights, axis=1, keepdims=False), num_xticks=5,
    #  title="b l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
    #  save_filename=(self.disp_dir+"b_norm_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      if name == "a":
        pf.plot_data_tiled(grad.T.reshape(self.num_u,
          int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
          normalize=False, title="Gradient for a at step "+current_step, vmin=None, vmax=None,
          save_filename=(self.disp_dir+"da_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
      #elif name == "b":
      #  pf.plot_data_tiled(grad.T.reshape(self.num_v,
      #    int(np.sqrt(self.num_u)), int(np.sqrt(self.num_u))),
      #    normalize=False, title="Gradient for b at step "+current_step, vmin=None, vmax=None,
      #    save_filename=(self.disp_dir+"db_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
