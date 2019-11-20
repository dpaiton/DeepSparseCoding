import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class RicaModel(Model):
  """
  Implementation of Quoc Le et al. Reconstruction ICA described in:
  QV Le, A Karpenko, J Ngiam, AY Ng (2011) - ICA with Reconstruction Cost for
  Efficient Overcomplete Feature Learning

  ## TODO:
  * rica model has different interface for applying gradients when the L-BFGS minimizer is used,
    which is inconsistent and should be changed so that it acts like all models
  """
  def __init__(self):
    super(RicaModel, self).__init__()
    self.vector_inputs = True

  def load_params(self, params):
    super(RicaModel, self).load_params(params)
    self.input_shape = [None, self.params.num_pixels]
    self.w_shape = [self.params.num_pixels, self.params.num_neurons]

  def compute_recon_from_encoding(self, a_in):
    return tf.matmul(a_in, tf.transpose(self.w), name="reconstruction")

  def compute_recon_loss(self, a_in):
    with tf.compat.v1.variable_scope("unsupervised"):
      recon_loss = tf.multiply(self.recon_mult,
        tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.compute_recon_from_encoding(a_in),
        self.input_placeholder)), axis=[1])), name="recon_loss")
    return recon_loss

  def compute_sparse_loss(self, a_in):
    with tf.compat.v1.variable_scope("unsupervised"):
      sparse_loss = tf.multiply(self.sparse_mult,
        tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.cosh(a_in)), axis=[1])), name="sparse_loss")
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

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.recon_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="recon_mult") # lambda
          self.sparse_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="sparse_mult")

        with tf.compat.v1.variable_scope("weights") as scope:
          w_init = tf.nn.l2_normalize(tf.random.truncated_normal(self.w_shape, mean=0.0, stddev=1.0),
            axis=[0], name="w_init")
          w_unnormalized = tf.compat.v1.get_variable(name="w", dtype=tf.float32, initializer=w_init,
            trainable=True)
          self.trainable_variables[w_unnormalized.name] = w_unnormalized
          w_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(w_unnormalized), axis=[0],
            keepdims=True), self.params.eps))
          self.w = tf.divide(w_unnormalized, w_norm, name="w_norm")

        with tf.compat.v1.variable_scope("inference") as scope:
          self.a = tf.matmul(input_node, self.w, name="activity")

        with tf.compat.v1.variable_scope("output") as scope:
          self.reconstruction = self.compute_recon_from_encoding(self.a)

        with tf.compat.v1.variable_scope("loss") as scope:
          loss_funcs = self.get_loss_funcs()
          self.loss_dict = dict(zip(
            [key for key in loss_funcs.keys()], [func(self.a) for func in loss_funcs.values()]))
          self.total_loss = self.compute_total_loss(self.a, loss_funcs)

        with tf.compat.v1.variable_scope("performance_metrics") as scope:
          with tf.compat.v1.variable_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.reconstruction)), axis=[1, 0],
              name="mean_squared_error")
            pixel_var = tf.nn.moments(input_node, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.math.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")
    self.graph_built = True

  def get_input_shape(self):
    return self.input_shape

  def get_num_latent(self):
    return self.params.num_neurons

  def get_encodings(self):
    return self.a

  def get_total_loss(self):
    return self.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Generates a dictionary to be logged in the print_update function
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(RicaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.loss_dict["recon_loss"], self.loss_dict["sparse_loss"],
      self.total_loss, self.a, self.reconstruction]
    if self.params.optimizer != "lbfgsb":
      eval_list.append(self.learning_rates)
      grad_name_list = []
      for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
        eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
        grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
        grad_name_list.append(grad_name)
    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
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
    if self.params.optimizer != "lbfgsb":
      lrs = out_vals[6]
      grads = out_vals[7:]
      for w_idx, (grad, name) in enumerate(zip(grads, grad_name_list)):
        grad_max = lrs[0][w_idx]*np.array(grad.max())
        grad_min = lrs[0][w_idx]*np.array(grad.min())
        grad_mean = lrs[0][w_idx]*np.mean(np.array(grad))
        stat_dict[name+"_lr"] = lrs[0][w_idx]
        stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(RicaModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w, self.reconstruction,  self.a]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    weights, recon, activity = eval_out[1:]
    #w_lengths = np.sqrt(np.sum(np.square(weights), axis=0))
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [units, pixels]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=self.params.disp_dir+"img_hist"+filename_suffix)
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Images at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"images"+filename_suffix)
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=self.params.disp_dir+"act_hist"+filename_suffix)
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"w"+filename_suffix)
    #fig = pf.plot_bar(w_lengths, title="Weight L2 Norms", xlabel="Weight Index", ylabel="L2 Norm",
    #  save_filename=self.params.disp_dir+"w_norms"+filename_suffix)
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)
    if self.params.optimizer != "lbfgsb":
      for weight_grad_var in self.grads_and_vars[self.sched_idx]:
        grad = weight_grad_var[0][0].eval(feed_dict)
        shape = grad.shape
        name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
        grad = dp.reshape_data(grad.T, flatten=False)[0]
        fig = pf.plot_data_tiled(grad, normalize=True,
          title="Gradient for w at step "+current_step, vmin=None, vmax=None,
          save_filename=self.params.disp_dir+"dw"+filename_suffix)
