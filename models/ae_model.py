import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.base_model import Model
from modules.ae_module import AeModule
from modules.activations import activation_picker

class AeModel(Model):
  def __init__(self):
    """
    Autoencoder
    """
    super(AeModel, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(AeModel, self).load_params(params)
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None, self.num_pixels]
    self.act_func = activation_picker(self.params.activation_function)

  def build_module(self):
    module = AeModule(self.x, self.params.output_channels, self.decay_mult, self.act_func,
      name="AE")
    return module

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")

        with tf.name_scope("placeholders") as sess:
          self.latent_input = tf.placeholder(tf.float32, name="latent_input")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.module = self.build_module()
        self.trainable_variables.update(self.module.trainable_variables)

        with tf.name_scope("inference") as scope:
          self.a = tf.identity(self.module.a, name="activity")

        # first index grabs u_list, second index grabs recon
        self.decoder_recon = self.module.build_decoder(self.module.num_encoder_layers+1,
          self.latent_input, [self.act_func,]*(self.module.num_decoder_layers-1)+[tf.identity],
          self.module.w_shapes[self.module.num_encoder_layers:])[0][-1]

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.module.reconstruction)),
              axis=[1, 0], name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")

  def compute_recon(self, a_in):
    #TODO: use self.decoder_recon with placeholder, fix analysis
    recon = self.module.build_decoder(self.module.num_encoder_layers+1,
      a_in, [self.act_func,]*(self.module.num_decoder_layers-1)+[tf.identity],
      self.module.w_shapes[self.module.num_encoder_layers:])[0][-1]
    return recon


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
      self.module.reconstruction, self.learning_rates]
    grad_name_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[-1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, decay_loss, total_loss, a_vals, recon = out_vals[0:6]
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
      "a_fraction_active":a_frac_act,
      "a_max_mean_min":[a_vals_max, a_vals_mean, a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
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
    super(AeModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w_list[0], self.module.w_list[-1],
      self.module.b_list, self.module.u_list[1:]]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, w_dec, b_list, activations = eval_out[1:]
    recon = activations[-1]
    # compute weight norms
    w_enc_norm = np.linalg.norm(w_enc, axis=0, keepdims=False)
    w_dec_norm = np.linalg.norm(w_dec, axis=1, keepdims=False)
    # reshapes flat data into image & normalize
    w_enc_img = dp.reshape_data(w_enc.T, flatten=False)[0]
    w_dec_img = dp.reshape_data(w_dec, flatten=False)[0]
    w_enc_img = dp.norm_weights(w_enc_img)
    w_dec_img = dp.norm_weights(w_dec_img)
    # generate figures
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    fig = pf.plot_data_tiled(w_enc_img, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_enc" + filename_suffix))
    fig = pf.plot_bar(w_enc_norm, num_xticks=5,
      title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_enc_norm"+filename_suffix))
    fig = pf.plot_data_tiled(w_dec_img, normalize=False,
      title="Decoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_dec" + filename_suffix))
    fig = pf.plot_bar(w_dec_norm, num_xticks=5,
      title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_dec_norm"+filename_suffix))
    for layer_id, activity in enumerate(activations[:-1]):
      fig = pf.plot_activity_hist(activity,
        title="Activity Encoder " + str(layer_id) + " Histogram",
        save_filename=(self.params.disp_dir+"act_enc_"+str(layer_id)+"_hist" + filename_suffix))
    for layer_id, bias in enumerate(b_list):
      fig = pf.plot_activity_hist(np.squeeze(bias),
        title="Bias " + str(layer_id) + " Histogram",
        save_filename=(self.params.disp_dir+"bias_"+str(layer_id)+"_hist" + filename_suffix))
    if eval_out[0]*10 % self.params.cp_int == 0:
      #Scale image by max and min of images and/or recon
      r_max = np.max([np.max(input_data), np.max(recon)])
      r_min = np.min([np.min(input_data), np.min(recon)])
      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.params.disp_dir+"img_hist" + filename_suffix))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.params.disp_dir+"images"+filename_suffix))
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.params.disp_dir+"recons"+filename_suffix))
