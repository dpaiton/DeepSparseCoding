import numpy as np
import tensorflow as tf

import DeepSparseCoding.utils.plot_functions as pf
import DeepSparseCoding.utils.data_processing as dp
from DeepSparseCoding.models.base_model import Model

class IcaModel(Model):
  def __init__(self):
    super(IcaModel, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(IcaModel, self).load_params(params)
    assert (True if self.params.prior.lower() in ("laplacian", "cauchy") else False), (
      "Prior must be 'laplacian' or 'cauchy'")
    ## Calculated params
    self.num_neurons = self.params.num_pixels
    self.params.num_neurons = self.num_neurons
    self.input_shape = [None, self.params.num_pixels]
    self.w_synth_shape = [self.num_neurons, self.params.num_pixels]
    self.w_analysis_shape = [self.params.num_pixels, self.num_neurons]

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("weights") as scope:
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
          self.w_synth = tf.compat.v1.get_variable(name="w_synth", dtype=tf.float32,
            initializer=Q.astype(np.float32), trainable=True)
          # w_analysis is W in Bell & Sejnowski 1997, which is used to compute the activations
          self.w_analysis = tf.linalg.inv(self.w_synth, name="w_analysis")
          self.trainable_variables[self.w_synth.name] = self.w_synth

          # Bell & Sejnowsky 1997
          #self.w_analysis = tf.compat.v1.get_variable(name="w_analysis", dtype=tf.float32,
          #  initializer=Q.astype(np.float32), trainable=True)
          #self.w_synth = tf.linalg.inv(self.w_analysis, name="w_synth")
          #self.trainable_variables[self.w_analysis.name] = self.w_analysis

        with tf.compat.v1.variable_scope("inference") as scope:
          self.a = tf.matmul(input_node, self.w_analysis, name="activity")
          if self.params.prior.lower() == "laplacian":
            self.z = tf.sign(self.a)
          else: #It must be laplacian or cauchy
            self.z = (2*self.a) / (1 + tf.pow(self.a, 2.0))

        with tf.compat.v1.variable_scope("output") as scope:
          self.reconstruction = tf.matmul(self.a, self.w_synth, name="reconstruction")

    self.graph_built = True

  def compute_weight_gradients(self, optimizer, weight_op=None):
    """
    Returns the natural gradient for the ICA weight matrix
    NOTE:
      This child function does not use optimizer input
      weight_op must be a list with a single matrix ("self.w_synth") in it
    """
    if(type(weight_op) is not list):
      weight_op = [weight_op]

    assert len(weight_op) == 1, ("IcaModel should only have one weight matrix")
    weight_name = weight_op[0].name.split('/')[1].split(':')[0]# last one is np.split

    # VS 265 solution
    # Note this is performed on w_synthesis (A in Bell & Sejnowski 1997), while the B&S paper gives
    # an update rule for W.
    z_a_avg = tf.math.divide(tf.matmul(tf.transpose(self.z), self.a),
      tf.cast(tf.shape(self.input_placeholder)[0], tf.float32), name="avg_samples")
    gradient = -tf.subtract(tf.matmul(tf.transpose(z_a_avg), weight_op[0]), weight_op[0],
      name=weight_name+"_gradient") # weight_op[0] is expected to be w_synth

    # Bell & Sejnowsky 1997
    #z_a_avg = tf.math.divide(tf.matmul(tf.transpose(self.a), self.z),
    #  tf.to_float(tf.shape(self.x)[0]), name="avg_samples")
    #gradient = -tf.add(tf.matmul(weight_op[0], z_a_avg), weight_op[0],
    #  name=weight_name+"_gradient") # weight_op[0] is expected to be w_analysis
    #gradient = -tf.add(weight_op[0], tf.multiply(self.z, tf.matmul(self.a, weight_op[0])
    return [(gradient, weight_op[0])]

  def compute_recon_from_placeholder(self):
    return self.reconstruction

  def get_input_shape(self):
    return self.input_shape

  def get_num_latent(self):
    return self.num_neurons

  def get_encodings(self):
    return self.a

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Generates a dictionary to be logged in the print_update function
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    NOTE: Casting tf.eval output to an np.array and then to a list is required to
      ensure that the data type is valid for js.dumps(). An alternative would be
      to write an np function that converts numpy types to their corresponding
      python types.
    """
    update_dict = super(IcaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list  = [self.global_step, self.a, self.z, self.reconstruction]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step, a_vals, z_vals, recon_data = out_vals[:4]
    input_max = np.max(input_data)
    input_mean = np.mean(input_data)
    input_min = np.min(input_data)
    recon_max = np.max(recon_data)
    recon_mean = np.mean(recon_data)
    recon_min = np.min(recon_data)
    a_vals_max = np.max(a_vals)
    a_vals_mean = np.mean(a_vals)
    a_vals_min = np.min(a_vals)
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(self.num_neurons * self.params.batch_size))
    z_vals_max = np.max(z_vals)
    z_vals_mean = np.mean(z_vals)
    z_vals_min = np.min(z_vals)
    z_frac_act = np.array(np.count_nonzero(z_vals)
      / float(self.num_neurons * self.params.batch_size))
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "number_of_batch_steps":self.get_schedule("num_batches"),
      "schedule_index":self.sched_idx,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "a_fraction_active":a_frac_act,
      "z_max_mean_min":[z_vals_max, z_vals_mean, z_vals_min],
      "z_fraction_active":z_frac_act,
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    grads = out_vals[4:]
    for grad, name in zip(grads, grad_name_list):
      grad_max = learning_rate_dict[name]*np.array(grad.max())
      grad_min = learning_rate_dict[name]*np.array(grad.min())
      grad_mean = learning_rate_dict[name]*np.mean(np.array(grad))
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
    update_dict.update(stat_dict) # stat_dict vals overwrite
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(IcaModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_analysis,  self.a, self.z]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, a_vals, z_vals = eval_out[1:]
    #input_data = dp.reshape_data(input_data, flatten=False)[0]
    #pf.plot_data_tiled(input_data, normalize=False,
    #  title="Images at step "+current_step, vmin=np.min(input_data), vmax=np.max(input_data),
    #  save_filename=(self.params.disp_dir+"images-"+current_step.zfill(5)+".png"))
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False) # norm across pixels
    pf.plot_bar(weights_norm, num_xticks=5,
      title="$W_{analysis}$ l$_{2}$ norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_analysis_norm_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    weights = dp.reshape_data(weights.T, flatten=False)[0] #[neurons, pixels_y, pixels_x]
    pf.plot_weights(weights.squeeze(), title="Unnormalized weights at step "+current_step,
      save_filename=(self.params.disp_dir+"w_analysis_unnormalized_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    #pf.plot_data_tiled(weights, normalize=True,
    #  title="Weights at step "+current_step, vmin=-1.0, vmax=1.0,
    #  save_filename=(self.params.disp_dir+"w_analysis_v"+self.params.version+"-"
    #  +current_step.zfill(5)+".png"))
    pf.plot_activity_hist(a_vals, num_bins=1000,
      title="a Activity Histogram at step "+current_step,
      save_filename=(self.params.disp_dir+"act_hist_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    #pf.plot_activity_hist(z_vals, num_bins=1000,
    #  title="z Activity Histogram at step "+current_step,
    #  save_filename=(self.params.disp_dir+"z_hist_v"+self.params.version+"-"
    #  +current_step.zfill(5)+".png"))
    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad, flatten=False)[0]
    #  pf.plot_data_tiled(grad, normalize=False,
    #    title="Gradient for "+name+" at step "+current_step, vmin=None, vmax=None,
    #    save_filename=(self.params.disp_dir+"d"+name+"_v"+self.params.version+"_"+current_step.zfill(5)+".png"))
