import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.base_model import Model
from modules.lca_module import LcaModule
import pdb

class LcaModel(Model):
  def __init__(self):
    super(LcaModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(LcaModel, self).load_params(params)

    # Network Size
    #self.input_shape = [None, self.params.num_pixels]
    self.input_shape = [None,] + self.params.data_shape

    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def get_input_shape(self):
    return self.input_shape

  def build_module(self, input_node):
    module = LcaModule(input_node, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.thresh_type, self.params.rectify_a,
      self.params.num_steps, self.params.eps)
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("auto_placeholders") as scope:
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")


        #Flatten input_node if not flat
        data_shape = input_node.get_shape().as_list()
        if len(data_shape) == 4:
          self.is_image = True
          self.batch_size, self.y_size, self.x_size, self.num_data_channels = data_shape
          self.num_pixels = self.y_size * self.x_size * self.num_data_channels
          input_node = tf.reshape(input_node, [-1, self.num_pixels])
        else:
          self.is_image = False

        self.input_node = input_node


        self.module = self.build_module(input_node)
        self.trainable_variables.update(self.module.trainable_variables)

        with tf.variable_scope("inference") as scope:
          self.a = tf.identity(self.get_encodings(), name="activity")

        with tf.variable_scope("placeholders") as sess:
          self.latent_input = tf.placeholder(tf.float32, shape=self.a.get_shape().as_list(),
            name="latent_input")

        with tf.variable_scope("norm_weights") as scope:
          self.norm_weights = tf.group(self.module.norm_w, name="l2_normalization")

        with tf.variable_scope("output") as scope:
          self.decoder_recon = self.module.build_decoder(self.latent_input, name="latent_recon")
          self.reconstruction = tf.identity(self.compute_recon_from_encoding(self.a),
            name="reconstruction")

        with tf.variable_scope("performance_metrics") as scope:
          MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.module.reconstruction)),
            name="mean_squared_error")
          pixel_var = tf.nn.moments(input_node, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var),
            MSE)), name="recon_quality")

  def compute_recon_from_placeholder(self):
    return self.decoder_recon

  def compute_recon_from_encoding(self, a_in):
    return self.module.build_decoder(a_in, name="reconstruction")

  def get_encodings(self):
    return self.module.a

  #TODO: change to have access to the module's total loss for inference analysis
  # Not sure what other dependencies there are on get_total_loss...
  def get_total_loss(self):
    return self.module.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(LcaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.loss_dict["recon_loss"],
      self.module.loss_dict["sparse_loss"], self.get_total_loss(), self.get_encodings(),
      self.module.reconstruction, self.pSNRdB, self.module.w, self.input_node]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name(1)]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] # 2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, a_vals, recon, pSNRdB, weights, input_node = out_vals[0:9]
    #input_max = np.max(input_node)
    #input_mean = np.mean(input_node)
    #input_min = np.min(input_node)
    input_max = np.max(input_node)
    input_mean = np.mean(input_node)
    input_min = np.min(input_node)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_mean = np.array(a_vals.mean())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    w_vals_min = weights.min()
    w_vals_mean = weights.mean()
    w_vals_max = weights.max()
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "sparse_loss":sparse_loss,
      "total_loss":total_loss,
      "pSNRdB": np.mean(pSNRdB),
      "a_fraction_active":a_frac_act,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min],
      "w_max_mean_min":[w_vals_max, w_vals_mean, w_vals_min]
      }
    grads = out_vals[8:]
    for grad, name in zip(grads, grad_name_list):
      grad_max = learning_rate_dict[name]*np.array(grad.max())
      grad_min = learning_rate_dict[name]*np.array(grad.min())
      grad_mean = learning_rate_dict[name]*np.mean(np.array(grad))
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
      stat_dict[name+"_learning_rate"] = learning_rate_dict[name]
    update_dict.update(stat_dict) #stat_dict overwrites for same keys
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(LcaModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w, self.module.reconstruction, self.get_encodings(), self.input_node]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    weights, recon, activity, input_node = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    if(self.is_image):
      num_batch = recon.shape[0]
      recon = dp.reshape_data(recon, out_shape=[num_batch, self.y_size, self.x_size, self.num_data_channels])[0]
      num_weights = weights.shape[-1]
      weights = dp.reshape_data(weights.T, out_shape=[num_weights, self.y_size, self.x_size, self.num_data_channels])[0] # [num_neurons, height, width]
      input_node = dp.reshape_data(input_node, out_shape=[num_batch, self.y_size, self.x_size, self.num_data_channels])[0]
    else:
      recon = dp.reshape_data(recon, flatten=False)[0]
      weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
      input_node = dp.reshape_data(input_node, flatten=False)[0]

    #fig = pf.plot_activity_hist(input_node, title="Image Histogram",
    #  save_filename=self.params.disp_dir+"img_hist"+filename_suffix)

    #Scale image by max and min of images and/or recon
    r_max = np.max([np.max(input_node), np.max(recon)])
    r_min = np.min([np.min(input_node), np.min(recon)])

    fig = pf.plot_data_tiled(input_node, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"images"+filename_suffix)
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)
    fig = pf.plot_activity_hist(activity, title="Activity Histogram",
      save_filename=self.params.disp_dir+"act_hist"+filename_suffix)
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)

    #Plot loss over time
    eval_list = [self.module.recon_loss_list, self.module.sparse_loss_list, self.module.total_loss_list]
    (recon_losses, sparse_losses, total_losses) = tf.get_default_session().run(eval_list, feed_dict)
    #TODO put this in plot functions
    pf.plot_sc_losses(recon_losses, sparse_losses, total_losses,
      save_filename=self.params.disp_dir+"losses"+filename_suffix)






    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad.T, flatten=False)[0]
    #  fig = pf.plot_data_tiled(grad, normalize=True,
    #    title="Gradient for w at step "+current_step, vmin=None, vmax=None,
    #    save_filename=self.params.disp_dir+"dphi"+filename_suffix)
