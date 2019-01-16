import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.base_model import Model
from modules.vae_module import VaeModule
from modules.mlp_module import MlpModule
from modules.activations import activation_picker

class MlpVaeModel(Model):
  def __init__(self):
    """
    MLP trained on the reconstruction from VAE
    """
    super(MlpVaeModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpVaeModel, self).load_params(params)
    # Network Size
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.input_shape = [None, self.num_pixels]
    self.label_shape = [None, self.params.num_classes]
    self.act_func = activation_picker(self.params.activation_function)

  def get_input_shape(self):
    return self.input_shape

  def build_vae_module(self, input_node):
    module = VaeModule(input_node, self.params.vae_output_channels, self.sparse_mult,
      self.decay_mult, self.kld_mult, self.act_func, self.params.noise_level, name="VAE")
    return module

  def build_mlp_module(self):
    if self.params.train_on_recon:
      if self.params.layer_types[0] == "conv":
        data_shape = [tf.shape(self.input_placeholder)[0]]+self.params.full_data_shape
        recon = tf.reshape(self.vae_module.reconstruction, shape=data_shape)
      elif self.params.layer_types[0] == "fc":
        recon = self.vae_module.reconstruction
      else:
        assert False, ("params.layer_types must be 'fc' or 'conv'")
      module = MlpModule(recon, self.label_placeholder, self.params.layer_types,
        self.params.mlp_output_channels, self.params.batch_norm, self.params.dropout,
        self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
        self.params.patch_size_y, self.params.patch_size_x, self.params.conv_strides,
        self.params.eps, loss_type="softmax_cross_entropy", name="MLP")
    else: # train on VAE latent encoding
      assert self.params.layer_types[0] == "fc", (
        "MLP must have FC layers to train on VAE activity")
      module = MlpModule(self.vae_module.a, self.label_placeholder, self.params.layer_types,
        self.params.output_channels, self.params.batch_norm, self.params.dropout,
        self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
        self.params.patch_size_y, self.params.patch_size_x, self.params.conv_strides,
        self.params.eps, loss_type="softmax_cross_entropy", name="MLP")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.label_placeholder = tf.placeholder(tf.float32,
            shape=self.label_shape, name="input_labels")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.kld_mult = tf.placeholder(tf.float32, shape=(), name="kld_mult")
          self.train_vae = tf.placeholder(tf.bool, shape=(), name="train_vae")

        self.train_vae = tf.cast(self.train_vae, tf.float32)

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        #TODO: with tf.name_scope("vae_module"):
        self.vae_module = self.build_vae_module(input_node)
        self.trainable_variables.update(self.vae_module.trainable_variables)
        #TODO: with tf.name_scope("mlp_module"):
        self.mlp_module = self.build_mlp_module()
        self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.name_scope("loss") as scope:
          #Loss switches based on train_vae flag
          self.total_loss = self.train_vae * self.vae_module.total_loss + \
            (1-self.train_vae) * self.mlp_module.total_loss

        self.label_est = self.mlp_module.label_est

        with tf.name_scope("performance_metrics") as scope:
          #VAE metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.vae_module.reconstruction)),
            axis=[1, 0], name="mean_squared_error")
          pixel_var = tf.nn.moments(input_node, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.divide(tf.square(pixel_var), MSE)),
            name="recon_quality")
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.label_est, axis=1),
              tf.argmax(self.label_placeholder, axis=1), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

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
    update_dict = super(MlpVaeModel, self).generate_update_dict(input_data, input_labels,
      batch_step)
    if self.get_schedule("train_vae"):
      feed_dict = self.get_feed_dict(input_data, input_labels)
      eval_list = [self.global_step, self.vae_module.loss_dict["recon_loss"],
        self.vae_module.loss_dict["sparse_loss"], self.get_total_loss(),
        self.vae_module.a, self.vae_module.reconstruction, self.pSNRdB]
      grad_name_list = []
      learning_rate_dict = {}
      for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
        eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name(1)]
        grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] # 2nd is np.split
        grad_name_list.append(grad_name)
        learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
      out_vals =  tf.get_default_session().run(eval_list, feed_dict)
      current_step, recon_loss, sparse_loss, total_loss, vae_a_vals, recon, pSNRdB\
        = out_vals[0:8]
      input_max = np.max(input_data)
      input_mean = np.mean(input_data)
      input_min = np.min(input_data)
      recon_max = np.max(recon)
      recon_mean = np.mean(recon)
      recon_min = np.min(recon)
      vae_a_vals_max = np.array(vae_a_vals.max())
      vae_a_vals_mean = np.array(vae_a_vals.mean())
      vae_a_vals_min = np.array(vae_a_vals.min())
      vae_a_frac_act = np.array(np.count_nonzero(vae_a_vals)
        / float(vae_a_vals.size))
      stat_dict = {"global_batch_index":current_step,
        "batch_step":batch_step,
        "schedule_index":self.sched_idx,
        "vae_recon_loss":recon_loss,
        "vae_sparse_loss":sparse_loss,
        "total_loss":total_loss,
        "pSNRdB": np.mean(pSNRdB),
        "vae_a_fraction_active":vae_a_frac_act,
        "vae_a_max_mean_min":[vae_a_vals_max, vae_a_vals_mean, vae_a_vals_min],
        "x_max_mean_min":[input_max, input_mean, input_min],
        "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
      grads = out_vals[7:]
      for grad, name in zip(grads, grad_name_list):
        grad_max = learning_rate_dict[name]*np.array(grad.max())
        grad_min = learning_rate_dict[name]*np.array(grad.min())
        grad_mean = learning_rate_dict[name]*np.mean(np.array(grad))
        stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]
        stat_dict[name+"_learning_rate"] = learning_rate_dict[name]
      update_dict.update(stat_dict) #stat_dict overwrites for same keys
    else:
      feed_dict = self.get_feed_dict(input_data, input_labels)
      current_step = np.array(self.global_step.eval())
      total_loss = np.array(self.get_total_loss().eval(feed_dict))
      logits_vals = tf.get_default_session().run(self.get_encodings(), feed_dict)
      logits_vals_max = np.array(logits_vals.max())
      logits_frac_act = np.array(np.count_nonzero(logits_vals) / float(logits_vals.size))
      accuracy = np.array(self.accuracy.eval(feed_dict))
      stat_dict = {"global_batch_index":current_step,
        "batch_step":batch_step,
        "number_of_batch_steps":self.params.schedule[self.sched_idx]["num_batches"],
        "schedule_index":self.sched_idx,
        "total_loss":total_loss,
        "logits_max":logits_vals_max,
        "logits_frac_active":logits_frac_act,
        "train_accuracy":accuracy}
      update_dict.update(stat_dict) #stat_dict overwrites
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(MlpVaeModel, self).generate_plots(input_data, input_labels)

    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.vae_module.w_list[0],
      self.vae_module.reconstruction, self.vae_module.a, self.get_encodings()]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    weights, recon, vae_activity, activity = eval_out[1:]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=(self.params.disp_dir+"img_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(recon, title="Recon Histogram",
      save_filename=(self.params.disp_dir+"recon_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
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
    fig = pf.plot_activity_hist(vae_activity, title="VAE Activity Histogram",
      save_filename=(self.params.disp_dir+"vae_act_hist" + name_suffix))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"phi" + name_suffix))
    fig = pf.plot_activity_hist(activity, title="Logit Histogram",
      save_filename=(self.params.disp_dir+"act_hist_"+self.params.version+"-"
      +current_step.zfill(5)+".png"))
