import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.entropy_functions as ef
from DeepSparseCoding.tf1x.models.ae_model import AeModel
from DeepSparseCoding.tf1x.modules.vae_module import VaeModule
from DeepSparseCoding.tf1x.modules.activations import activation_picker

class VaeModel(AeModel):
  def __init__(self):
    """
    Variational Autoencoder using Mixture of Gaussians
    DP Kingma and M Welling (2014) - "Auto-encoding variational bayes."
    https://arxiv.org/pdf/1312.6114.pdf

    Variational sparse coding
    F Tonolini, BS Jensen, R Murray-Smith (2018) - Variational Sparse Coding
    https://openreview.net/pdf?id=SkeJ6iR9Km

    Sparse Coding Variational Autoencoders
    G Barello, AS Charles, JW Pillow (2018) - Sparse-Coding Variational Auto-Encoders
    https://www.biorxiv.org/content/biorxiv/early/2018/08/29/399246.full.pdf
    """
    super(VaeModel, self).__init__() # will call super.load_params() and super.ae_load_params()

  def load_params(self, params):
    super(VaeModel, self).load_params(params)
    self.vae_load_params(params) # only vae-specific params

  def vae_load_params(self, params):
    self.vae_mean_act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.vae_mean_activation_functions]
    self.vae_var_act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.vae_var_activation_functions]
    self.num_latent = self.params.vae_mean_channels[-1]

  def build_module(self, input_node):
    module = VaeModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.vae_mean_act_funcs, self.params.vae_mean_layer_types,
      self.params.vae_mean_channels, self.params.vae_mean_patch_size,
      self.params.vae_mean_conv_strides, self.params.vae_mean_dropout,
      self.vae_var_act_funcs, self.params.vae_var_layer_types, self.params.vae_var_channels,
      self.params.vae_var_patch_size, self.params.vae_var_conv_strides, self.params.vae_var_dropout,
      self.w_decay_mult, self.w_norm_mult, self.kld_mult, self.act_funcs,
      self.ae_dropout_keep_probs, self.params.tie_dec_weights, self.params.noise_level,
      self.params.prior_params, self.params.w_init_type,
      variable_scope="vae")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.kld_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="kld_mult")
    super(VaeModel, self).build_graph_from_input(input_node)

  def get_encodings(self):
    return self.module.act

  def get_total_loss(self):
    return self.module.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(VaeModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    latent_loss =  tf.compat.v1.get_default_session().run(self.module.loss_dict["latent_loss"], feed_dict)
    stat_dict = {"latent_loss":latent_loss}
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(VaeModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.act]
    if self.params.noise_level > 0.0:
      eval_list += [self.module.corrupt_data]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    latent_code = eval_out[1]
    if self.params.noise_level > 0.0:
      corrupt_data  = eval_out[2]
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    #TODO histogram with large bins is broken
    #fig = pf.plot_activity_hist(b_enc_std, title="Encoding Bias Std Histogram",
    #  save_filename=(self.disp_dir+"b_enc_std_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    latent_layer = self.module.num_enc_layers-1
    if self.params.noise_level > 0.0:
      corrupt_data = dp.reshape_data(corrupt_data, flatten=False)[0]
      fig = pf.plot_data_tiled(corrupt_data, normalize=False,
        title="Corrupted Images at step "+current_step,
        save_filename=self.params.disp_dir+"corrupt_images"+filename_suffix)
    # Plot generated digits
    latent_shape = latent_code.shape[1:]
    randoms = [np.random.normal(0, 1, latent_shape) for _ in range(self.params.batch_size)]
    feed_dict[self.latent_input] = np.stack(randoms, axis=0)
    feed_dict[self.ae_dropout_keep_probs] = [1.0] * len(self.params.ae_dropout)
    imgs = tf.compat.v1.get_default_session().run(self.compute_recon_from_placeholder(), feed_dict)
    imgs = imgs.reshape(imgs.shape[0], self.params.num_edge_pixels, self.params.num_edge_pixels,
      self.params.num_data_channels)
    #if imgs.ndim == 2:
    #  imgs = np.stack([np.reshape(imgs[i], [28, 28, 1]) for i in range(len(imgs))], axis=0)
    imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
    gen_imgs_fig = pf.plot_data_tiled(imgs, normalize=False,
      title="Generated images", vmin=0, vmax=1,
      save_filename=(self.params.disp_dir+"generated_images"
      +"_v"+self.params.version+"_"+str(current_step).zfill(5)+".png"))
    if self.params.ae_layer_types[-1] == "fc":
      # display a 30x30 2D manifold of digits
      n = 30
      digit_size = int(np.sqrt(self.params.num_pixels))
      figure_img = np.zeros((digit_size * n, digit_size * n))
      # linearly spaced coordinates corresponding to the 2D plot
      # of digit classes in the latent space
      grid_x = np.linspace(-4, 4, n)
      grid_y = np.linspace(-4, 4, n)[::-1]
      num_z = self.num_latent
      for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]+[0.0]*(num_z-2)])
          feed_dict[self.latent_input] = z_sample
          x_decoded = tf.compat.v1.get_default_session().run(self.compute_recon_from_placeholder(),
            feed_dict)
          digit = x_decoded[0].reshape(digit_size, digit_size)
          figure_img[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
      fig, ax = plt.subplots(1, figsize=(10, 10))
      start_range = digit_size // 2
      end_range = n * digit_size + start_range + 1
      pixel_range = np.arange(start_range, end_range, digit_size)
      sample_range_x = np.round(grid_x, 1)
      sample_range_y = np.round(grid_y, 1)
      ax.set_xticks(pixel_range)
      ax.set_xticklabels(sample_range_x)
      ax.set_yticks(pixel_range)
      ax.set_yticklabels(sample_range_y)
      ax.set_xlabel("latent[0]", fontsize=16)
      ax.set_ylabel("latent[1]", fontsize=16)
      ax.set_title("Generated Images from Latent Interpolation", fontsize=16)
      ax.imshow(figure_img, cmap='Greys_r')
      fig.savefig(self.params.disp_dir+"generated_latent_interpolation"+filename_suffix)
      plt.close(fig)
    if input_labels is not None and self.params.ae_layer_types[-1] == "fc":
      z_mean = tf.compat.v1.get_default_session().run(self.get_encodings(), feed_dict)
      fig, ax = plt.subplots(1, figsize=(12, 10))
      sc = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=dp.one_hot_to_dense(input_labels))
      fig.colorbar(sc)
      ax.set_xlabel("latent[0]", fontsize=16)
      ax.set_ylabel("latent[1]", fontsize=16)
      ax.set_title("Latent Encoding of Labeled Examples", fontsize=16)
      fig.savefig(self.params.disp_dir+"latent_enc"+filename_suffix)
      plt.close(fig)
