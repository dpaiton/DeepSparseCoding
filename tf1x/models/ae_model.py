import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.entropy_functions as ef
from DeepSparseCoding.tf1x.models.base_model import Model
from DeepSparseCoding.tf1x.modules.ae_module import AeModule
from DeepSparseCoding.tf1x.modules.activations import activation_picker

class AeModel(Model):
  def __init__(self):
    """
    Autoencoder
    """
    super(AeModel, self).__init__()

  def load_params(self, params):
    super(AeModel, self).load_params(params)
    self.ae_load_params(params)

  def ae_load_params(self, params):
    self.input_shape = [None,] + self.params.data_shape
    if self.params.mirror_dec_architecture: # Mirror each param as if the full amount is not specified
      num_enc_layers = len(self.params.ae_enc_channels)
      self.params.ae_activation_functions = self.params.ae_activation_functions[:num_enc_layers]
      self.params.ae_activation_functions += self.params.ae_activation_functions[::-1]
      self.params.ae_layer_types = self.params.ae_layer_types[:num_enc_layers]
      self.params.ae_layer_types += self.params.ae_layer_types[::-1]
      if("conv" in self.params.ae_layer_types):
        self.params.ae_conv_strides = self.params.ae_conv_strides[:num_enc_layers]
        self.params.ae_conv_strides += self.params.ae_conv_strides[::-1]
        self.params.ae_patch_size = self.params.ae_patch_size[:num_enc_layers]
        self.params.ae_patch_size += self.params.ae_patch_size[::-1]
      self.params.ae_dec_channels = self.params.ae_enc_channels[::-1][1:]
      self.params.ae_dec_channels += [self.params.data_shape[-1]]
      self.params.ae_dropout = self.params.ae_dropout[:num_enc_layers]
      self.params.ae_dropout += self.params.ae_dropout[::-1]
    self.num_latent = self.params.ae_enc_channels[-1]
    self.act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.ae_activation_functions]
    if np.all([layer_type == "fc" for layer_type in self.params.ae_layer_types]):
      self.params.ae_patch_size = []
      self.params.ae_conv_strides = []
    assert len(self.params.ae_dropout) == len(self.params.ae_activation_functions), \
        ("Dropout parameter must be a list of size " + str(len(self.params.ae_activation_functions)))

  def build_module(self, input_node):
    module = AeModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.w_decay_mult, self.w_norm_mult, self.act_funcs, self.ae_dropout_keep_probs,
      self.params.tie_dec_weights, self.params.w_init_type, variable_scope="ae")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.w_decay_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="w_decay_mult")
          self.w_norm_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="w_norm_mult")

        with tf.compat.v1.variable_scope("placeholders") as scope:
          self.ae_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="ae_dropout_keep_probs")
          self.latent_input = tf.compat.v1.placeholder(tf.float32, name="latent_input")

        self.module = self.build_module(input_node)
        self.trainable_variables.update(self.module.trainable_variables)

        with tf.compat.v1.variable_scope("norm_weights") as scope:
          self.norm_weights = tf.group(self.module.norm_w, name="l2_normalization")

        with tf.compat.v1.variable_scope("inference") as scope:
          self.a = tf.identity(self.module.a, name="activity")

        with tf.compat.v1.variable_scope("output") as scope:
          self.reconstruction = tf.identity(self.module.reconstruction, name="reconstruction")

        # first index grabs u_list, second index grabs recon
        #Need to build this in same namescope as the orig decoder
        with tf.compat.v1.variable_scope(self.module.variable_scope):
          self.dec_recon = self.module.build_decoder(self.latent_input,
            self.act_funcs[self.module.num_enc_layers:])[0][-1]

        with tf.compat.v1.variable_scope("performance_metrics") as scope:
          with tf.compat.v1.variable_scope("reconstruction_quality"):
            recon_shape = self.module.reconstruction.get_shape()
            data_shape = input_node.get_shape()
            if(recon_shape.ndims != data_shape.ndims):
              if(np.prod(recon_shape.as_list()[1:]) == np.prod(data_shape.as_list()[1:])):
                self.MSE = tf.reduce_mean(input_tensor=tf.square(tf.subtract(input_node,
                  tf.reshape(self.module.reconstruction, tf.shape(input=input_node)))),
                  axis=[1, 0], name="mean_sauared_error")
              else:
                assert False, ("Reconstructiion and input must have the same size")
            else:
              self.MSE = tf.reduce_mean(input_tensor=tf.square(tf.subtract(input_node,
                self.module.reconstruction)), axis=[1, 0], name="mean_squared_error")
            pixel_var = tf.nn.moments(x=input_node, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.math.log(tf.math.divide(tf.square(pixel_var),
              self.MSE)), name="recon_quality")

  def compute_recon_from_placeholder(self):
    return self.dec_recon

  def compute_recon_from_encoding(self, a_in):
    with tf.compat.v1.variable_scope(self.module.variable_scope):
      recon = self.module.build_decoder(a_in, self.act_funcs[self.module.num_enc_layers:])[0][-1]
    return recon

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    feed_dict = super(AeModel, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
    if(is_test): # Turn off dropout when not training
      feed_dict[self.ae_dropout_keep_probs] = [1.0,] * len(self.params.ae_dropout)
    else:
      feed_dict[self.ae_dropout_keep_probs] = self.params.ae_dropout
    return feed_dict

  def get_input_shape(self):
    return self.input_shape

  def get_num_latent(self):
    return self.num_latent

  def get_encodings(self):
    return self.module.a

  def get_total_loss(self):
    return self.module.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(AeModel, self).generate_update_dict(input_data,
      input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.module.loss_dict["recon_loss"],
      self.module.loss_dict["weight_decay_loss"], self.get_total_loss(), self.get_encodings(),
      self.module.reconstruction, self.MSE, self.learning_rates]

    grad_name_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[-1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)

    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, decay_loss, total_loss, a_vals, recon, mse = out_vals[0:7]
    mse_mean = np.mean(mse)
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_vals_mean = np.mean(a_vals)
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))

    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "total_loss":total_loss,
      "mse_mean":mse_mean,
      "a_fraction_active":a_frac_act,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}

    lrs = out_vals[7]
    grads = out_vals[8:]
    for w_idx, (grad, name) in enumerate(zip(grads, grad_name_list)):
      grad_max = lrs[0][w_idx]*np.array(grad.max())
      grad_min = lrs[0][w_idx]*np.array(grad.min())
      grad_mean = lrs[0][w_idx]*np.mean(np.array(grad))
      stat_dict[name+"_lr"] = lrs[0][w_idx]
      stat_dict[name+"_grad_max_mean_min"] = [grad_max, grad_mean, grad_min]

    names = []
    tf_vars = []
    for var_name, var in self.trainable_variables.items():
      names.append(var_name)
      tf_vars.append(var)

    np_vars = tf.compat.v1.get_default_session().run(tf_vars, feed_dict)
    for var_name, np_v in zip(names, np_vars):
      v_max = np.max(np_v)
      v_min = np.min(np_v)
      v_mean = np.mean(np_v)
      stat_dict[var_name+"_max_mean_min"] = [v_max, v_mean, v_min]

      num_f = np_v.shape[-1]
      v_reshape = np.reshape(np_v, [-1, num_f])
      v_norm = np.linalg.norm(np_v, axis=0)
      mean_v_norm = np.mean(v_norm)
      stat_dict[var_name+"_norm"] = mean_v_norm

    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(AeModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w_list[0], self.module.w_list[-1],
      self.module.b_list, self.module.u_list[1:]]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, w_dec, b_list, activations = eval_out[1:]
    recon = activations[-1]
    # compute weight norms
    num_features = w_enc.shape[-1]
    w_enc_norm = np.linalg.norm(np.reshape(w_enc, (-1, num_features)), axis=0, keepdims=False)
    # reshapes flat data into image & normalize
    if(len(w_enc.shape) == 2):
      w_enc_img = dp.reshape_data(w_enc.T, flatten=False)[0]
    else:
      w_enc_img = np.transpose(w_enc, (3, 0, 1, 2))
    w_enc_img = dp.norm_weights(w_enc_img)

    if(not self.params.tie_dec_weights):
      if(len(w_dec.shape) == 2):
        w_dec_norm = np.linalg.norm(w_dec, axis=1, keepdims=False)
        w_dec_img = dp.reshape_data(w_dec, flatten=False)[0]
      else:
        #Decoder in same shape as encoder if multi dimensional
        #conv2d_transpose requires it to be the same shape as decoder
        w_dec_norm = np.linalg.norm(np.reshape(w_dec, (-1, num_features)), axis=0, keepdims=False)
        w_dec_img = np.transpose(w_dec, (3, 0, 1, 2))
      w_dec_img = dp.norm_weights(w_dec_img)

    # generate figures
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"

    fig = pf.plot_data_tiled(w_enc_img, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"w_enc"+filename_suffix)

    fig = pf.plot_bar(w_enc_norm, num_xticks=5,
      title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=self.params.disp_dir+"w_enc_norm"+filename_suffix)

    if(not self.params.tie_dec_weights):
      fig = pf.plot_data_tiled(w_dec_img, normalize=False,
        title="Decoding weights at step "+current_step, vmin=None, vmax=None,
        save_filename=self.params.disp_dir+"w_dec"+filename_suffix)
      fig = pf.plot_bar(w_dec_norm, num_xticks=5,
        title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=self.params.disp_dir+"w_dec_norm"+filename_suffix)

    for layer_id, activity in enumerate(activations[:-1]):
      num_features = activity.shape[-1]
      fig = pf.plot_activity_hist(np.reshape(activity, (-1, num_features)),
        title="Activity Encoder " + str(layer_id) + " Histogram",
        save_filename=self.params.disp_dir+"act_enc_"+str(layer_id)+"_hist"+filename_suffix)

    for layer_id, bias in enumerate(b_list):
      fig = pf.plot_activity_hist(np.squeeze(bias),
        title="Bias " + str(layer_id) + " Histogram",
        save_filename=self.params.disp_dir+"bias_"+str(layer_id)+"_hist"+filename_suffix)
    if eval_out[0]*10 % self.params.cp_int == 0:
      #Scale image by max and min of images and/or recon
      r_max = np.max([np.max(input_data), np.max(recon)])
      r_min = np.min([np.min(input_data), np.min(recon)])
      batch_size = input_data.shape[0]
      fig = pf.plot_activity_hist(np.reshape(input_data, (batch_size, -1)),
        title="Image Histogram", save_filename=self.params.disp_dir+"img_hist"+filename_suffix)
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=self.params.disp_dir+"images"+filename_suffix)
      #TODO: This plot hangs sometimes?
      #fig = pf.plot_activity_hist(recon, title="Recon Histogram",
        #save_filename=self.params.disp_dir+"recon_hist"+filename_suffix)
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=self.params.disp_dir+"recons"+filename_suffix)
