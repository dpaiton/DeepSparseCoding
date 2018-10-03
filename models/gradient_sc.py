import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class Gradient_SC(Model):
  def __init__(self):
    super(Gradient_SC, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      batch_size   [int] Number of images in a training batch
      num_neurons  [int] Number of SC neurons
      num_a_steps    [int] Number of inference steps
    """
    super(Gradient_SC, self).load_params(params)
    self.data_shape = params["data_shape"]
    # Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    self.num_neurons = int(params["num_neurons"])
    self.phi_shape = [self.num_pixels, self.num_neurons]
    self.x_shape = [None, self.num_pixels]
    # Hyper Parameters
    self.num_a_steps = int(params["num_a_steps"])
    self.a_step_size = float(params["a_step_size"])

  def step_inference(self, a_in, loss, step):
    with tf.name_scope("update_a"+str(step)) as scope:
      da = tf.gradients(loss, a_in, name="deda")[0]
      a_out = tf.add(a_in, tf.multiply(self.a_step_size, -da))
    return a_out

  def infer_coefficients(self):
   a_list = [tf.matmul(self.x, self.phi, name="init_a")]
   loss_list = [self.compute_total_loss(a_list[0], self.get_loss_funcs())]
   for step in range(self.num_a_steps-1):
     a = self.step_inference(a_list[step], loss_list[step], step)
     loss = self.compute_total_loss(a, self.get_loss_funcs())
     a_list.append(a)
     loss_list.append(loss)
   return a_list

  def compute_recon(self, a_in):
    return tf.matmul(a_in, tf.transpose(self.phi), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.reduce_mean(0.5 *
        tf.reduce_sum(tf.square(tf.subtract(self.x, self.compute_recon(a_in))),
        axis=reduc_dim), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      sparse_loss = self.sparse_mult * tf.reduce_mean(tf.reduce_sum(tf.abs(a_in),
        axis=reduc_dim), name="sparse_loss")
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
    super(Gradient_SC, self).build_graph()
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          phi_init = tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=0.5, dtype=tf.float32, name="phi_init")
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
    Generates a dictionary to be logged in the print_update function
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.total_loss, self.a]
    grad_name_list = []
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name_list.append(weight_grad_var[0][1].name.split('/')[1].split(':')[0])#2nd is np.split
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, a_vals = out_vals[0:5]
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
      "sparse_loss":sparse_loss,
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
    return stat_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(Gradient_SC, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.phi, self.x_,  self.a]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, activity = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=1, keepdims=False)
    weights = dp.reshape_data(weights.T, flatten=False)[0]
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"phi_v"+self.version+"_"
      +current_step.zfill(5)+".png"))
    if eval_out[0]*10 % self.cp_int == 0:
      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
        +current_step.zfill(5)+".png"))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Images at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"images_"+self.version+"-"
        +current_step.zfill(5)+".png"))
      fig = pf.plot_activity_hist(activity, title="Activity Histogram",
        save_filename=(self.disp_dir+"act_hist_v"+self.version+"-"
        +current_step.zfill(5)+".png"))
      fig = pf.plot_bar(weights_norm, num_xticks=5,
        title="phi l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=(self.disp_dir+"phi_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
      recon = dp.reshape_data(recon, flatten=False)[0]
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
