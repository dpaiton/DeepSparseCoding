import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
import ops.init_ops as init_ops
from models.base_model import Model
from modules.lca_module import LcaModule
from modules.mlp_module import MlpModule

class ListaModel(Model):
  def __init__(self):
    super(ListaModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(ListaModel, self).load_params(params)
    # Network Size
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None, self.num_pixels]
    self.w_shape = [self.num_pixels, self.params.num_neurons]
    self.s_shape = [self.params.num_neurons, self.params.num_neurons]
    # Hyper Parameters
    self.eta = self.params.dt / self.params.tau

  def build_lca_module(self):
    module = LcaModule(self.x, self.params.num_neurons, self.sparse_mult,
      self.eta, self.params.thresh_type, self.params.rectify_a,
      self.params.num_steps, self.params.eps, name="lca")
    return module

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.train_lca = tf.placeholder(tf.bool, shape=(), name="train_lca")

        self.train_lca = tf.cast(self.train_lca, tf.float32)

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.lca_module = self.build_lca_module()
        self.trainable_variables.update(self.lca_module.trainable_variables)

        with tf.name_scope("weight_inits") as scope:
          self.w_init = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)
          self.s_init = init_ops.GDNGammaInitializer(diagonal_gain=0.0, off_diagonal_gain=0.001,
            dtype=tf.float32)

        with tf.variable_scope("weights") as scope:
          self.w = tf.get_variable(name="w_enc", shape=self.w_shape, dtype=tf.float32,
            initializer=self.w_init, trainable=True)
          self.s = tf.get_variable(name="lateral_connectivity", shape=self.s_shape,
            dtype=tf.float32, initializer=self.s_init, trainable=True)
        self.trainable_variables.update({self.w.name:self.w, self.s.name:self.s})

        with tf.name_scope("inference") as scope:
          feedforward_drive = tf.matmul(self.x, self.w, name="feedforward_drive")
          self.a_list = [self.lca_module.threshold_units(feedforward_drive, name="a_init")]
          for layer_id in range(self.params.num_layers):
            self.a_list.append(self.lca_module.threshold_units(feedforward_drive
              + tf.matmul(self.a_list[layer_id], self.s)))
          self.a = self.a_list[-1]

        with tf.name_scope("loss") as scope:
          reduc_dim = list(range(1, len(self.lca_module.a.shape)))
          labels = tf.stop_gradient(self.lca_module.a)
          self.lista_loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - self.a),
            axis=reduc_dim))
          #Loss switches based on train_lca flag
          self.total_loss = self.train_lca * self.lca_module.total_loss + \
            (1-self.train_lca) * self.lista_loss

        with tf.name_scope("norm_weights") as scope:
          self.norm_weights = tf.group(self.lca_module.norm_w, name="l2_normalization")

        with tf.name_scope("performance_metrics") as scope:
          #LCA metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.lca_module.reconstruction)),
            axis=[1, 0], name="mean_squared_error")
          pixel_var = tf.nn.moments(self.x, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var),
            MSE)), name="recon_quality")

  def get_encodings(self):
    return self.a

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
    update_dict = super(ListaModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.lca_module.loss_dict["recon_loss"],
      self.lca_module.loss_dict["sparse_loss"], self.get_total_loss(),
      self.lca_module.a, self.get_encodings(), self.lca_module.reconstruction, self.pSNRdB,
      self.lista_loss]
    grad_name_list = []
    learning_rate_dict = {}
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name(1)]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] # 2nd is np.split
      grad_name_list.append(grad_name)
      learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, sparse_loss, total_loss, lca_a_vals, lista_a_vals, recon, pSNRdB, lista_loss \
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
      "lista_loss":lista_loss,
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
    super(ListaModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.lca_module.w,
      self.lca_module.reconstruction, self.lca_module.a, self.get_encodings()]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, lca_activity, lista_activity = eval_out[1:]
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.params.disp_dir+"img_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))

    #Scale image by max and min of images and/or recon
    r_max = np.max([np.max(input_data), np.max(recon)])
    r_min = np.min([np.min(input_data), np.min(recon)])

    name_suffix = "_v"+self.params.version+"-"+current_step.zfill(5)+".png"

    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=(self.params.disp_dir+"images" + name_suffix))
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=(self.params.disp_dir+"recons" + name_suffix))

    fig = pf.plot_activity_hist(lca_activity, title="LCA Activity Histogram",
      save_filename=(self.params.disp_dir+"lca_act_hist" + name_suffix))

    fig = pf.plot_activity_hist(lista_activity, title="LISTA Activity Histogram",
      save_filename=(self.params.disp_dir+"lista_act_hist" + name_suffix))

    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"phi" + name_suffix))

    #for weight_grad_var in self.grads_and_vars[self.sched_idx]:
    #  grad = weight_grad_var[0][0].eval(feed_dict)
    #  shape = grad.shape
    #  name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]#np.split
    #  grad = dp.reshape_data(grad.T, flatten=False)[0]
    #  fig = pf.plot_data_tiled(grad, normalize=True,
    #    title="Gradient for w at step "+current_step, vmin=None, vmax=None,
    #    save_filename=(self.params.disp_dir+"dphi" + name_suffix))
