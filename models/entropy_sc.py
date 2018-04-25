import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
from models.gradient_sc import Gradient_SC

class Entropy_SC(Gradient_SC):
  def __init__(self):
    super(Entropy_SC, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      mle_step_size: [float]
      num_mle_steps: [int]
      num_triangles: [int]
      sigmoid_beta: [float]
    """
    super(Entropy_SC, self).load_params(params)
    # Hyper Parameters
    self.mle_step_size = float(params["mle_step_size"])
    self.num_mle_steps = int(params["num_mle_steps"])
    self.num_triangles = int(params["num_triangles"])
    self.sigmoid_beta = float(params["sigmoid_beta"])

  def sigmoid(self, a_in, beta=1):
    """Sigmoidal non-linearity"""
    a_out = tf.subtract(tf.multiply(2.0, tf.divide(1.0,
      tf.add(1.0, tf.exp(tf.multiply(-beta, a_in))))), 1.0)
    return a_out

  def compute_entropy_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      a_sig = self.sigmoid(a_in, self.sigmoid_beta)
      a_probs = ef.prob_est(a_sig, self.mle_thetas, self.triangle_centers)
      a_entropies = tf.identity(ef.calc_entropy(a_probs), name="a_entropies")
      entropy_loss = tf.multiply(self.entropy_mult, tf.reduce_sum(a_entropies), name="entropy_loss")
    return entropy_loss

  def get_loss_funcs(self):
    return {"recon_loss":self.compute_recon_loss, "entropy_loss":self.compute_entropy_loss}

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.triangle_centers = tf.placeholder(tf.float32, shape=[self.num_triangles],
            name="triangle_centers")
          self.entropy_mult = tf.placeholder(tf.float32, shape=(), name="entropy_mult")

        with tf.variable_scope("probability_estimate") as scope:
          self.mle_thetas, self.theta_init = ef.construct_thetas(self.num_neurons,
            self.num_triangles)

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          phi_init = tf.nn.l2_normalize(tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=0.5, dtype=tf.float32, name="phi_init"), epsilon=self.eps, name="row_l2_norm")
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=phi_init, trainable=True)

        with tf.name_scope("norm_weights") as scope:
          phi_norm_dim = list(range(len(self.phi_shape)-1)) # normalize across input dim(s)
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, dim=phi_norm_dim,
            epsilon=self.eps, name="row_l2_norm"))
          self.norm_weights = tf.group(self.norm_phi, name="l2_normalization")

        with tf.variable_scope("inference") as scope:
         a_list = self.infer_coefficients()
         self.a = tf.identity(a_list[-1], name="activity")

        with tf.variable_scope("probability_estimate") as scope:
          self.a_sig = self.sigmoid(self.a, self.sigmoid_beta)
          ll = ef.log_likelihood(self.a_sig, self.mle_thetas, self.triangle_centers)
          self.mle_update = [ef.mle(ll, self.mle_thetas, self.mle_step_size)
            for _ in range(self.num_mle_steps)]

        with tf.name_scope("loss") as scope:
          loss_funcs = self.get_loss_funcs()
          self.loss_dict = dict(zip(
            [key for key in loss_funcs.keys()], [func(self.a) for func in loss_funcs.values()]))
          self.total_loss = self.compute_total_loss(self.a, loss_funcs)

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = self.compute_recon(self.a)

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
    super(Entropy_SC, self).print_update(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["entropy_loss"],
      self.total_loss, self.a]
    grad_name_list = []
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name_list.append(weight_grad_var[0][1].name.split('/')[1].split(':')[0])#2nd is np.split
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, entropy_loss, total_loss, a_vals = out_vals[0:5]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "entropy_loss":entropy_loss,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_min":a_vals_min,
      "a_fraction_active":a_frac_act,
      "x_mean":input_mean,
      "x_max":input_max,
      "x_min":input_min}
    grads = out_vals[5:]
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
    super(Entropy_SC, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.phi, self.x_,  self.a]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, activity = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=1, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    #fig = pf.plot_data_tiled(input_data, normalize=False,
    #  title="Images at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"images_"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=(self.disp_dir+"act_hist_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"
      +current_step.zfill(5)+".png"))
    #fig = pf.plot_bar(weights_norm, num_xticks=5,
    #  title="phi l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
    #  save_filename=(self.disp_dir+"phi_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      grad = dp.reshape_data(grad.T, flatten=False)[0]
      fig = pf.plot_data_tiled(grad, normalize=True,
        title="Gradient for phi at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"dphi_v"+self.version+"_"+current_step.zfill(5)+".png"))
