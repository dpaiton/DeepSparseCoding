import numpy as np
import logging
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class ICA(Model):
  def __init__(self, params, schedule):
    super(ICA, self).__init__(params, schedule)

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      prior        [str] Prior for ICA - can be "laplacian" or "cauchy"
      batch_size   [int] Number of images in a training batch
      num_pixels   [int] Number of pixels
    """
    super(ICA, self).load_params(params)
    ## Meta parameters
    self.prior = str(params["prior"])
    assert (True if self.prior.lower() in ("laplacian", "cauchy") else False), (
      "Prior must be 'laplacian' or 'cauchy'")
    ## Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.patch_edge_size = int(params["patch_edge_size"])
    self.num_patch_pixels = int(self.patch_edge_size**2)
    self.num_neurons = self.num_patch_pixels
    self.a_shape = [self.num_neurons, self.num_patch_pixels]

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(
            tf.float32, shape=[None, self.num_pixels], name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          ## Q matrix from QR decomp is guaranteed to be orthonormal and
          ## non-singular, which prevents a gradient explosion from inverting
          ## the weight matrix.
          ## NOTE: TF does not currently have a stable QR decomp function
          ##  Issue: https://github.com/tensorflow/tensorflow/issues/4739
          ##  Commit: ""/commit/715f951eb9ca20fdcef20bb544b74dbe576734da
          #rand = tf.truncated_normal(self.a_shape, mean=0.0,
          #  stddev=1.0, dtype=tf.float32, name="stand_norm_dist")
          #Q, R = tf.qr(rand, full_matrices=True, name="qr_decomp")
          Q, R = np.linalg.qr(np.random.standard_normal(self.a_shape))
          self.a = tf.get_variable(name="a", dtype=tf.float32,
            initializer=Q.astype(np.float32), trainable=True)
          self.a_inv = tf.matrix_inverse(self.a, name="a_inverse")

        with tf.name_scope("inference") as scope:
          self.u = tf.matmul(self.x, self.a_inv, name="coefficients")
          if self.prior.lower() == "laplacian":
            self.z = tf.sign(self.u)
          else: #It must be laplacian or cauchy
            self.z = (2*self.u) / (1 + tf.pow(self.u, 2.0))

    self.graph_built = True

  def compute_weight_gradients(self, optimizer, weight_op=None):
    """
    Returns the natural gradient for the ICA weight matrix
    NOTE:
      This child function does not use optimizer input
      weight_op must be a list with a single matrix ("self.a") in it
    """
    assert len(weight_op) == 1, ("ICA should only have one weight matrix")
    weight_name = weight_op[0].name.split('/')[1].split(':')[0]#np.split
    z_u_avg = tf.divide(tf.matmul(tf.transpose(self.u), self.z),
      tf.to_float(tf.shape(self.x)[0]), name="avg_samples")
    gradient = -tf.subtract(tf.matmul(z_u_avg, weight_op[0]), weight_op[0],
      name=weight_name+"_gradient")
    return [(gradient, weight_op[0])]

  def print_update(self, input_data, input_labels=None, batch_step=0):
    """
    Logs progress information
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    NOTE: Casting tf.eval output to an np.array and then to a list is required to
      ensure that the data type is valid for js.dumps(). An alternative would be
      to write an np function that converts numpy types to their corresponding
      python types.
    """
    super(ICA, self).print_update(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval()).tolist()
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    u_vals_max = np.array(u_vals.max()).tolist()
    u_frac_act = np.array(np.count_nonzero(u_vals)
      / float(self.num_neurons * self.batch_size)).tolist()
    z_vals = tf.get_default_session().run(self.z, feed_dict)
    z_vals_max = np.array(z_vals.max()).tolist()
    z_frac_act = np.array(np.count_nonzero(z_vals)
      / float(self.num_neurons * self.batch_size)).tolist()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_sched("num_batches"),
      "schedule_index":self.sched_idx,
      "u_max":u_vals_max,
      "u_fraction_active":u_frac_act,
      "z_max":z_vals_max,
      "z_fraction_active":z_frac_act}
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      stat_dict[name+"_max_grad"] = np.array(grad.max()).tolist()
      stat_dict[name+"_min_grad"] = np.array(grad.min()).tolist()
    js_str = js.dumps(stat_dict, sort_keys=True, indent=2)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(ICA, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    weights = tf.get_default_session().run(self.a, feed_dict)
    current_step = str(self.global_step.eval())
    #pf.plot_data_tiled(input_data.reshape((self.batch_size,
    #  np.int(np.sqrt(self.num_pixels)),
    #  np.int(np.sqrt(self.num_pixels)))),
    #  normalize=False, title="Images at step "+current_step,
    #  vmin=np.min(input_data), vmax=np.max(input_data),
    #  save_filename=(self.disp_dir+"images_"+current_step.zfill(5)+".pdf"))
    pf.plot_data_tiled(weights.reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step, vmin=-1.0, vmax=1.0,
      save_filename=(self.disp_dir+"a_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    pf.plot_activity_hist(self.z.eval(feed_dict), num_bins=1000,
      title="z Activity Histogram at step "+current_step,
      save_filename=(self.disp_dir+"z_hist_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    pf.plot_activity_hist(self.u.eval(feed_dict), num_bins=1000,
      title="u Activity Histogram at step "+current_step,
      save_filename=(self.disp_dir+"u_hist_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    pf.plot_bar(np.linalg.norm(weights, axis=1, keepdims=False), num_xticks=5,
      title="a l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"a_norm_v"+self.version+"-"+current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      pf.plot_data_tiled(grad.reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=False, title="Gradient for "+name+" at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"d"+name+"_v"+self.version+"_"+current_step.zfill(5)+".pdf"))
