import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class RICA(Model):
  """
  Implementation of Quoc Le et al. Reconstruction ICA described in:
  QV Le, A Karpenko, J Ngiam, AY Ng (2011)
  ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning
  """
  def __init__(self):
    super(RICA, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    super(RICA, self).load_params(params)
    self.data_shape = params["data_shape"]
    ## Network Size
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(np.prod(self.data_shape))
    self.num_neurons = int(params["num_neurons"])
    self.x_shape = [None, self.num_pixels]
    self.w_shape = [self.num_pixels, self.num_neurons]

  def compute_recon(self, a_in):
    return tf.matmul(a_in, tf.transpose(self.w), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      recon_loss = tf.multiply(self.recon_mult,
        tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.compute_recon(a_in), self.x)),
        axis=reduc_dim)), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      reduc_dim = list(range(1, len(a_in.shape))) # Want to avg over batch, sum over the rest
      sparse_loss = tf.reduce_mean(tf.reduce_sum(tf.log(tf.cosh(a_in)),
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
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.recon_mult = tf.placeholder(tf.float32, shape=(), name="recon_mult") # lambda

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          w_init = tf.truncated_normal(self.w_shape, mean=0.0, stddev=0.3,
            dtype=tf.float32, name="w_init")
          #self.w = tf.get_variable(name="w", dtype=tf.float32, initializer=w_init, trainable=True)
          w_unconstrained = tf.get_variable(name="w", dtype=tf.float32,
            initializer=w_init, trainable=True)
          self.w = tf.nn.l2_normalize(w_unconstrained, dim=0, epsilon=self.eps, name="w_norm")

        with tf.name_scope("inference") as scope:
          self.a = tf.matmul(self.x, self.w, name="activity")

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
    super(RICA, self).print_update(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.total_loss, self.a, self.x_]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, a_vals, recon = out_vals[0:6]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_mean = np.array(a_vals.mean())
    a_vals_min = np.array(a_vals.min())
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    grads = out_vals[6:]
    for grad, name in zip(grads, grad_name_list):
      grad_max = learning_rate_dict[name]*np.array(grad.max())
      grad_min = learning_rate_dict[name]*np.array(grad.min())
      grad_mean = learning_rate_dict[name]*np.mean(np.array(grad))
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
    js_str = self.js_dumpstring(stat_dict)
    self.log_info("<stats>"+js_str+"</stats>")

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(RICA, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w, self.x_,  self.a]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, activity = eval_out[1:]
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"images_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=(self.disp_dir+"act_hist_"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"w_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
    for weight_grad_var in self.grads_and_vars[self.sched_idx]:
      grad = weight_grad_var[0][0].eval(feed_dict)
      shape = grad.shape
      name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
      grad = dp.reshape_data(grad.T, flatten=False)[0]
      fig = pf.plot_data_tiled(grad, normalize=True,
        title="Gradient for w at step "+current_step, vmin=None, vmax=None,
        save_filename=(self.disp_dir+"dw_v"+self.version+"_"+current_step.zfill(5)+".png"))
