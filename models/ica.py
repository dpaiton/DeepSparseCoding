import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class ICA(Model):
  def __init__(self):
    super(ICA, self).__init__()
    self.vector_inputs = True

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
    self.data_shape = params["data_shape"]
    ## Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    self.num_neurons = self.num_pixels
    self.x_shape = [None, self.num_pixels]
    self.w_synth_shape = [self.num_neurons, self.num_pixels]
    self.w_analysis_shape = [self.num_pixels, self.num_neurons]

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          ## Q matrix from QR decomp is guaranteed to be orthonormal and
          ## non-singular, which prevents a gradient explosion from inverting
          ## the weight matrix.
          ## NOTE: TF does not currently have a stable QR decomp function
          ##  Issue: https://github.com/tensorflow/tensorflow/issues/4739
          ##  Commit: ""/commit/715f951eb9ca20fdcef20bb544b74dbe576734da
          #rand = tf.truncated_normal(self.w_synth_shape, mean=0.0,
          #  stddev=1.0, dtype=tf.float32, name="stand_norm_dist")
          #Q, R = tf.qr(rand, full_matrices=True, name="qr_decomp")
          Q, R = np.linalg.qr(np.random.standard_normal(self.w_analysis_shape))

          # VS 265 solution
          # w_synth (synthesis) is A in Bell & Sejnowski 1997, which are the basis functions
          self.w_synth = tf.get_variable(name="w_synth", dtype=tf.float32,
            initializer=Q.astype(np.float32), trainable=True)
          # w_analysis is W in Bell & Sejnowski 1997, which is used to compute the activations
          self.w_analysis = tf.matrix_inverse(self.w_synth, name="w_analysis")

          # Bell & Sejnowsky 1997
          #self.w_analysis = tf.get_variable(name="w_analysis", dtype=tf.float32,
          #  initializer=Q.astype(np.float32), trainable=True)
          #self.w_synth = tf.matrix_inverse(self.w_analysis, name="w_synth")

        with tf.name_scope("inference") as scope:
          self.a = tf.matmul(self.x, self.w_analysis, name="activity")
          if self.prior.lower() == "laplacian":
            self.z = tf.sign(self.a)
          else: #It must be laplacian or cauchy
            self.z = (2*self.a) / (1 + tf.pow(self.a, 2.0))

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.x_ = tf.matmul(self.a, self.w_synth, name="reconstruction")

    self.graph_built = True

  def compute_weight_gradients(self, optimizer, weight_op=None):
    """
    Returns the natural gradient for the ICA weight matrix
    NOTE:
      This child function does not use optimizer input
      weight_op must be a list with a single matrix ("self.w_synth") in it
    """
    assert len(weight_op) == 1, ("ICA should only have one weight matrix")
    weight_name = weight_op[0].name.split('/')[1].split(':')[0]# last one is np.split

    # VS 265 solution
    # Note this is performed on w_synthesis (A in Bell & Sejnowski 1997), while the B&S paper gives
    # an update rule for W.
    z_a_avg = tf.divide(tf.matmul(tf.transpose(self.z), self.a),
      tf.to_float(tf.shape(self.x)[0]), name="avg_samples")
    gradient = -tf.subtract(tf.matmul(tf.transpose(z_a_avg), weight_op[0]), weight_op[0],
      name=weight_name+"_gradient") # weight_op[0] is expected to be w_synth

    # Bell & Sejnowsky 1997
    #z_a_avg = tf.divide(tf.matmul(tf.transpose(self.a), self.z),
    #  tf.to_float(tf.shape(self.x)[0]), name="avg_samples")
    #gradient = -tf.add(tf.matmul(weight_op[0], z_a_avg), weight_op[0],
    #  name=weight_name+"_gradient") # weight_op[0] is expected to be w_analysis
    #gradient = -tf.add(weight_op[0], tf.multiply(self.z, tf.matmul(self.a, weight_op[0])
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
    current_step = np.array(self.global_step.eval())
    a_vals = tf.get_default_session().run(self.a, feed_dict)
    a_vals_max = np.array(a_vals.max())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.num_neurons * self.batch_size))
    z_vals = tf.get_default_session().run(self.z, feed_dict)
    z_vals_max = np.array(z_vals.max())
    z_frac_act = np.array(np.count_nonzero(z_vals)
      / float(self.num_neurons * self.batch_size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_schedule("num_batches"),
      "schedule_index":self.sched_idx,
      "a_max":a_vals_max,
      "a_fraction_active":a_frac_act,
      "z_max":z_vals_max,
      "z_fraction_active":z_frac_act}
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
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
    super(ICA, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_analysis,  self.a, self.z]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, a_vals, z_vals = eval_out[1:]
    #input_data = dp.reshape_data(input_data, flatten=False)[0]
    #pf.plot_data_tiled(input_data, normalize=False,
    #  title="Images at step "+current_step, vmin=np.min(input_data), vmax=np.max(input_data),
    #  save_filename=(self.disp_dir+"images-"+current_step.zfill(5)+".png"))
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False) # norm across pixels
    pf.plot_bar(weights_norm, num_xticks=5,
      title="$W_{analysis}$ l$_{2}$ norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.disp_dir+"w_analysis_norm_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    weights = dp.reshape_data(weights.T, flatten=False)[0] #[neurons, pixels_y, pixels_x]
    pf.plot_weights(weights.squeeze(), title="Unnormalized weights at step "+current_step,
      save_filename=(self.disp_dir+"w_analysis_unnormalized_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    #pf.plot_data_tiled(weights, normalize=True,
    #  title="Weights at step "+current_step, vmin=-1.0, vmax=1.0,
    #  save_filename=(self.disp_dir+"w_analysis_v"+self.version+"-"+current_step.zfill(5)+".png"))
    pf.plot_activity_hist(a_vals, num_bins=1000,
      title="a Activity Histogram at step "+current_step,
      save_filename=(self.disp_dir+"act_hist_v"+self.version+"-"+current_step.zfill(5)+".png"))
    #pf.plot_activity_hist(z_vals, num_bins=1000,
    #  title="z Activity Histogram at step "+current_step,
    #  save_filename=(self.disp_dir+"z_hist_v"+self.version+"-"+current_step.zfill(5)+".png"))
    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad, flatten=False)[0]
    #  pf.plot_data_tiled(grad, normalize=False,
    #    title="Gradient for "+name+" at step "+current_step, vmin=None, vmax=None,
    #    save_filename=(self.disp_dir+"d"+name+"_v"+self.version+"_"+current_step.zfill(5)+".png"))
