import numpy as np
import logging
import json as js
import tensorflow as tf
import utils.plot_functions as pf
from models.base_model import Model

class ICA(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)
    self.build_graph()
    Model.setup_graph(self, self.graph)

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  Modifiable Parameters:
    norm_weights [bool] If set, l2 normalize weights after updates
    prior        [str] Prior for ICA - can be "laplacian" or "cauchy"
    batch_size   [int] Number of images in a training batch
    num_pixels   [int] Number of pixels
  """
  def load_params(self, params):
    Model.load_params(self, params)
    # Meta parameters
    self.norm_weights = bool(params["norm_weights"])
    self.prior = str(params["prior"])
    assert (True if self.prior.lower() in ("laplacian", "cauchy") else False), (
      "Prior must be 'laplacian' or 'cauchy'")
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_neurons = self.num_pixels
    self.a_shape = [self.num_pixels, self.num_neurons]

  """Check parameters with assertions"""
  def check_params(self):
    Model.check_params(self)
    assert np.sqrt(self.num_pixels) == np.floor(np.sqrt(self.num_pixels)), (
      "The parameter `num_pixels` must have an even square-root.")

  """Build the TensorFlow graph object"""
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.x = tf.placeholder(
            tf.float32, shape=[self.num_pixels, None], name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.a = tf.get_variable(name="a", dtype=tf.float32,
            initializer=tf.truncated_normal(self.a_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="a_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_a = self.a.assign(tf.nn.l2_normalize(self.a,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_a,
            name="l2_normalization")

        with tf.name_scope("inference") as scope:
          self.u = tf.matmul(tf.matrix_inverse(self.a, name="a_inverse"),
            self.x, name="coefficients")
          if self.prior.lower() == "laplacian":
            self.z = tf.sign(self.u)
          else: #It must be laplacian or cauchy, assert in load_params()
            self.z = (2*self.u) / (1 + tf.pow(self.u, 2.0))

    self.graph_built = True

  """
    Returns the gradients for a weight variable
  NOTE:
    This child function does not use optimizer input
    Weights must be a list with a single matrix ("a") in it
  """
  def compute_gradients(self, optimizer, weight_op=None):
    assert len(weight_op) == 1, ("ICA should only have one weight matrix")
    z_avg = tf.div(tf.matmul(self.z, tf.transpose(self.u)),
      tf.to_float(tf.shape(self.x)[1]), name="avg_samples")
    weight_name = weight_op[0].name.split('/')[1].split(':')[0]
    gradient = tf.sub(tf.matmul(weight_op[0], z_avg), weight_op[0],
      name=weight_name+"_gradient")
    return [(gradient, weight_op[0])]

  """
    input_data: data object containing the current image batch
    input_labels: data object containing the current label batch
    batch_step: current batch number within the schedule
  NOTE: Casting tf.eval output to an np.array and then to a list is required to
    ensure that the data type is valid for js.dumps(). An alternative would be
    to write an np function that converts numpy types to their corresponding
    python types.
  """
  def print_update(self, input_data, input_labels=None, batch_step=0):
    Model.print_update(self, input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    current_step = np.array(self.global_step.eval()).tolist()
    u_vals = tf.get_default_session().run(self.u, feed_dict)
    u_vals_max = np.array(u_vals.max()).tolist()
    u_frac_act = np.array(np.count_nonzero(u_vals)
      / float(self.num_neurons * self.batch_size)).tolist()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_sched("num_batches"),
      "schedule_index":self.sched_idx,
      "u_max":u_vals_max,
      "u_fraction_active":u_frac_act}
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
    pf.save_data_tiled(input_data.T.reshape((self.batch_size,
      np.int(np.sqrt(self.num_pixels)),
      np.int(np.sqrt(self.num_pixels)))),
      normalize=False, title="Images at step "+current_step,
      save_filename=(self.disp_dir+"images_"
      +current_step.zfill(5)+".pdf"),
      vmin=np.min(input_data), vmax=np.max(input_data))
    pf.save_data_tiled(
      tf.transpose(self.a).eval().reshape(self.num_neurons,
      int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
      normalize=True, title="Dictionary at step "+current_step,
      save_filename=(self.disp_dir+"a_v"+self.version+"-"
      +current_step.zfill(5)+".pdf"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
      pf.save_data_tiled(grad.T.reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Gradient for "+name+" at step "+current_step,
        save_filename=(self.disp_dir+"d"+name+"_v"+self.version+"_"
        +current_step.zfill(5)+".pdf"))
