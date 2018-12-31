import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_funcs as ef
from models.base_model import Model
from modules.vae import VAE as vae_module
import pdb
class VAE(Model):
  def __init__(self):
    """
    Variational Autoencoder using Mixture of Gaussians
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
    arXiv preprint arXiv:1312.6114 (2013).
    """
    super(VAE, self).__init__()

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(VAE, self).load_params(params)
    self.num_pixels = int(np.prod(self.params.data_shape))
    self.x_shape = [None, self.num_pixels]

  def compute_recon(self, a_in):
    return self.vae.build_decoder(a_in)[-1]

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.decay_mult = tf.placeholder(tf.float32, shape=(), name="decay_mult")
          self.kld_mult = tf.placeholder(tf.float32, shape=(), name="kld_mult")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.name_scope("model") as scope:
          self.vae = vae_module(self.x, self.params.output_channels, self.sparse_mult,
              self.decay_mult, self.kld_mult, name="VAE")

          #Store losses here as member variables for analyzers
          self.loss_dict = self.vae.loss_dict
          self.total_loss = self.vae.total_loss

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.square(tf.subtract(self.x, self.vae.reconstruction)),
              axis=[1, 0], name="mean_squared_error")
            pixel_var = tf.nn.moments(self.x, axes=[1])[1]
            self.pSNRdB = tf.multiply(10.0, tf.log(tf.divide(tf.square(pixel_var), MSE)),
              name="recon_quality")


    self.graph_built = True

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(VAE, self).generate_update_dict(input_data,
      input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.vae.loss_dict["recon_loss"],
      self.vae.loss_dict["latent_loss"], self.vae.loss_dict["weight_decay_loss"],
      self.vae.total_loss, self.vae.a, self.vae.reconstruction, self.learning_rates]
    grad_name_list = []
    for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
      eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
      grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
      grad_name_list.append(grad_name)
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, latent_loss, decay_loss, total_loss, a_vals, recon = out_vals[0:7]
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
      "latent_loss":latent_loss,
      "total_loss":total_loss,
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
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(VAE, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.vae.w_enc_list, self.vae.w_dec_list, self.vae.w_enc_std,
      self.vae.b_enc_list, self.vae.b_enc_std, self.vae.b_dec_list,
      self.vae.encoder_activations, self.vae.decoder_activations, self.vae.a]

    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc, w_dec, w_enc_std, b_enc, b_enc_std, b_dec, enc_act, dec_act, a = eval_out[1:]
    recon = dec_act[-1]

    w_enc_norm = [np.linalg.norm(w, axis=0, keepdims=False) for w in w_enc]
    w_enc_std_norm = np.linalg.norm(w_enc_std, axis=0, keepdims=False)
    w_dec_norm = [np.linalg.norm(w, axis=1, keepdims=False) for w in w_dec]

    #Reshapes flat data into image
    w_enc_img = dp.reshape_data(w_enc[0].T, flatten=False)[0]
    w_dec_img = dp.reshape_data(w_dec[-1], flatten=False)[0]

    w_enc_img = dp.norm_weights(w_enc_img)
    w_dec_img = dp.norm_weights(w_dec_img)

    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"

    fig = pf.plot_data_tiled(w_enc_img, normalize=False,
      title="Encoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_enc" + filename_suffix))
    fig = pf.plot_data_tiled(w_dec_img, normalize=False,
      title="Decoding weights at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"w_dec" + filename_suffix))

    #TODO histogram with large bins is broken
    #fig = pf.plot_activity_hist(b_enc_mean, title="Encoding Bias Mean Histogram",
    #  save_filename=(self.disp_dir+"b_enc_mean_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #fig = pf.plot_activity_hist(b_enc_std, title="Encoding Bias Std Histogram",
    #  save_filename=(self.disp_dir+"b_enc_std_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    for l in range(len(enc_act)):
      fig = pf.plot_activity_hist(enc_act[l], title="Activity Encoder " + str(l) + " Histogram",
        save_filename=(self.params.disp_dir+"act_enc_"+str(l)+"_hist" + filename_suffix))
      fig = pf.plot_bar(w_enc_norm[l], num_xticks=5,
        title="w_enc_"+str(l)+" l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=(self.params.disp_dir+"w_enc_"+str(l)+"_norm"+filename_suffix))

      fig = pf.plot_activity_hist(dec_act[l], title="Activity Decoder " + str(l) + " Histogram",
        save_filename=(self.params.disp_dir+"act_dec_"+str(l)+"_hist" + filename_suffix))
      fig = pf.plot_bar(w_dec_norm[l], num_xticks=5,
        title="w_dec_"+str(l)+" l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
        save_filename=(self.params.disp_dir+"w_dec_"+str(l)+"_norm"+filename_suffix))

    fig = pf.plot_activity_hist(w_enc_std, title="Activity Encoder " + str(l) + " Std Histogram",
      save_filename=(self.params.disp_dir+"act_enc_"+str(l)+"_std_hist" + filename_suffix))
    fig = pf.plot_bar(w_enc_std_norm, num_xticks=5,
      title="w_enc_"+str(l)+"_std l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_enc_"+str(l)+"_std_norm" + filename_suffix))

    if eval_out[0]*10 % self.params.cp_int == 0:
      #Scale image by max and min of images and/or recon
      r_max = np.max([np.max(input_data), np.max(recon)])
      r_min = np.min([np.min(input_data), np.min(recon)])

      fig = pf.plot_activity_hist(input_data, title="Image Histogram",
        save_filename=(self.params.disp_dir+"img_hist" + filename_suffix))
      input_data = dp.reshape_data(input_data, flatten=False)[0]
      fig = pf.plot_data_tiled(input_data, normalize=False,
        title="Images at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.params.disp_dir+"images"+filename_suffix))
      recon = dp.reshape_data(recon, flatten=False)[0]
      fig = pf.plot_data_tiled(recon, normalize=False,
        title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
        save_filename=(self.params.disp_dir+"recons"+filename_suffix))
