import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.base_model import Model
from modules.lca_module import LcaModule
from modules.mlp_module import MlpModule

class FfListaModel(Model):
  def __init__(self):
    super(FfListaModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(FfListaModel, self).load_params(params)
    # Network Size
    self.input_shape = [None, self.params.num_pixels]
    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def get_input_shape(self):
    return self.input_shape

  def build_lca_module(self, input_node):
    module = LcaModule(input_node, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.thresh_type, self.params.rectify_a,
      self.params.num_steps, self.params.eps)
    return module

  def build_mlp_module(self, input_node):
    module = MlpModule(input_node, self.lca_module.a, self.params.layer_types,
      self.params.output_channels, self.params.batch_norm, self.params.dropout,
      self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
      self.params.patch_size_y, self.params.patch_size_x, self.params.conv_strides,
      self.params.eps, loss_type="l2")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.variable_scope("auto_placeholders") as scope:
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.train_lca = tf.placeholder(tf.bool, shape=(), name="train_lca")

        self.train_lca = tf.cast(self.train_lca, tf.float32)

        self.lca_module = self.build_lca_module(input_node)
        self.trainable_variables.update(self.lca_module.trainable_variables)
        self.mlp_module = self.build_mlp_module(input_node)
        self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.variable_scope("loss") as scope:
          #Loss switches based on train_lca flag
          self.total_loss = self.train_lca * self.lca_module.total_loss + \
            (1-self.train_lca) * self.mlp_module.total_loss

        with tf.variable_scope("norm_weights") as scope:
          self.norm_weights = tf.group(self.lca_module.norm_w, name="l2_normalization")

        with tf.variable_scope("performance_metrics") as scope:
          #LCA metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.lca_module.reconstruction)),
            axis=[1, 0], name="mean_squared_error")
          pixel_var = tf.nn.moments(input_node, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var),
            MSE)), name="recon_quality")

  def get_encodings(self):
    return self.mlp_module.layer_list[-1]

  def get_total_loss(self):
    return self.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Log train progress information
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(FfListaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.lca_module.loss_dict["recon_loss"],
      self.lca_module.loss_dict["sparse_loss"], self.get_total_loss(),
      self.lca_module.a, self.get_encodings(), self.lca_module.reconstruction, self.pSNRdB,
      self.mlp_module.total_loss]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name(1)]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] # 2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, lca_a_vals, lista_a_vals, recon, pSNRdB, fflista_loss \
      = out_vals[0:9]
    input_max = np.max(input_data)
    input_mean = np.mean(input_data)
    input_min = np.min(input_data)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    lca_a_vals_max = np.array(lca_a_vals.max())
    lca_a_vals_mean = np.array(lca_a_vals.mean())
    lca_a_vals_min = np.array(lca_a_vals.min())
    lca_a_frac_act = np.array(np.count_nonzero(lca_a_vals)
      / float(lca_a_vals.size))
    lista_a_vals_max = np.array(lista_a_vals.max())
    lista_a_vals_mean = np.array(lista_a_vals.mean())
    lista_a_vals_min = np.array(lista_a_vals.min())
    lista_a_frac_act = np.array(np.count_nonzero(lista_a_vals)
      / float(lista_a_vals.size))

    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "lca_recon_loss":recon_loss,
      "lca_sparse_loss":sparse_loss,
      "fflista_loss":fflista_loss,
      "total_loss":total_loss,
      "pSNRdB": np.mean(pSNRdB),
      "lca_a_fraction_active":lca_a_frac_act,
      "lca_a_max_mean_min":[lca_a_vals_max, lca_a_vals_mean, lca_a_vals_min],
      "lista_a_fraction_active":lista_a_frac_act,
      "lista_a_max_mean_min":[lista_a_vals_max, lista_a_vals_mean, lista_a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    grads = out_vals[9:]
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
    super(FfListaModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.lca_module.w,
      self.lca_module.reconstruction, self.lca_module.a, self.get_encodings()]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"

    weights, recon, lca_activity, lista_activity = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=self.params.disp_dir+"img_hist"+filename_suffix)

    #Scale image by max and min of images and/or recon
    r_max = np.max([np.max(input_data), np.max(recon)])
    r_min = np.min([np.min(input_data), np.min(recon)])

    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"images"+filename_suffix)
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=self.params.disp_dir+"recons"+filename_suffix)

    fig = pf.plot_activity_hist(lca_activity, title="LCA Activity Histogram",
      save_filename=self.params.disp_dir+"lca_act_hist"+filename_suffix)

    fig = pf.plot_activity_hist(lista_activity, title="LISTA Activity Histogram",
      save_filename=self.params.disp_dir+"lista_act_hist"+filename_suffix)

    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)

    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad.T, flatten=False)[0]
    #  fig = pf.plot_data_tiled(grad, normalize=True,
    #    title="Gradient for w at step "+current_step, vmin=None, vmax=None,
    #    save_filename=self.params.disp_dir+"dphi"+filename_suffix)
