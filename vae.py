"""
Implementation of VAE as described here:
https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.base_model import Model
from utils.logger import Logger
from params.base_params import BaseParams
import data.data_selector as ds
import utils.data_processing as dp
import utils.plot_functions as pf
from modules.activations import activation_picker

class VAE(Model):
  def load_params(self, params):
    super(VAE, self).load_params(params)
    self.input_shape = [28, 28, 1]
    self.n_latent = self.params.output_channels[-1]

  def get_input_shape(self):
    return [None]+self.input_shape

  def encoder(self, X_in):
    with tf.variable_scope("encoder", reuse=None):
      X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
      x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same',
        activation=activation_picker(self.params.activation_functions[0]))
      x = tf.nn.dropout(x, self.dropout_keep_probs[0])
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same',
        activation=activation_picker(self.params.activation_functions[1]))
      x = tf.nn.dropout(x, self.dropout_keep_probs[1])
      x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same',
        activation=activation_picker(self.params.activation_functions[2]))
      x = tf.nn.dropout(x, self.dropout_keep_probs[2])
      x = tf.contrib.layers.flatten(x)
      mn = tf.layers.dense(x, units=self.n_latent)
      sd = 0.5 * tf.layers.dense(x, units=self.n_latent)
      epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
      z = mn + tf.multiply(epsilon, tf.exp(sd))
      return z, mn, sd

  def decoder(self, sampled_z):
    with tf.variable_scope("decoder", reuse=None):
      dec_in_channels = 1
      inputs_decoder = 49 * dec_in_channels // 2
      x = tf.layers.dense(sampled_z, units=inputs_decoder,
        activation=activation_picker(self.params.activation_functions[3]))
      x = tf.layers.dense(x, units=inputs_decoder * 2 + 1,
        activation=activation_picker(self.params.activation_functions[4]))
      reshaped_dim = [-1, 7, 7, dec_in_channels]
      x = tf.reshape(x, reshaped_dim)
      x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4,
        strides=2, padding='same',
        activation=activation_picker(self.params.activation_functions[5]))
      x = tf.nn.dropout(x, self.dropout_keep_probs[3])
      x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1,
        padding='same',
        activation=activation_picker(self.params.activation_functions[6]))
      x = tf.nn.dropout(x, self.dropout_keep_probs[4])
      x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1,
        padding='same',
        activation=activation_picker(self.params.activation_functions[7]))
      x = tf.contrib.layers.flatten(x)
      x = tf.layers.dense(x, units=28*28,
        activation=activation_picker(self.params.activation_functions[8]))
      img = tf.reshape(x, shape=[-1, 28, 28])
      return img

  def build_graph_from_input(self, input_node):
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.Y = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_shape, name="Y")
          Y_flat = tf.reshape(self.Y, shape=[-1, 28*28])
          self.dropout_keep_probs = tf.placeholder(tf.float32, shape=[None],
            name="dropout_keep_probs")

        with tf.variable_scope("inference") as scope:
          self.sampled, self.mn, self.sd = self.encoder(input_node)

        with tf.variable_scope("outputs") as scope:
          self.dec = self.decoder(self.sampled)

        for var in tf.trainable_variables():
          self.trainable_variables[var.name] = var

        with tf.name_scope("loss") as scope:
          unreshaped = tf.reshape(self.dec, [-1, 28*28])
          self.img_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), axis=1))
          self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), axis=1))
          self.total_loss = self.img_loss + self.latent_loss

  def get_encodings(self):
    return self.logits

  def get_total_loss(self):
    return self.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    update_dict = super(VAE, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels, is_test=True)
    feed_dict[self.Y] = input_data
    eval_list = [self.global_step, self.img_loss, self.latent_loss, self.total_loss]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step, recon_loss, latent_loss, total_loss = out_vals
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "recon_loss":recon_loss,
      "latent_loss":latent_loss,
      "total_loss":total_loss}
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
    feed_dict = self.get_feed_dict(input_data, input_labels, is_test=True)
    feed_dict[self.Y] = input_data
    w_enc = self.graph.get_tensor_by_name("inference/encoder/conv2d/kernel:0")
    eval_list = [self.global_step, self.dec, w_enc]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    recon = eval_out[1][...,None]
    r_max = np.max([np.max(input_data), np.max(recon)])
    r_min = np.min([np.min(input_data), np.min(recon)])
    fig = pf.plot_activity_hist(input_data.reshape(-1, 28*28), title="Image Histogram",
      save_filename=(self.params.disp_dir+"img_hist" + filename_suffix))
    input_data = dp.reshape_data(input_data, flatten=False)[0]
    fig = pf.plot_data_tiled(input_data, normalize=False,
      title="Scaled Images at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=(self.params.disp_dir+"images"+filename_suffix))
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=r_min, vmax=r_max,
      save_filename=(self.params.disp_dir+"recons"+filename_suffix))
    weights = eval_out[2].transpose((3,0,1,2))
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.params.disp_dir+"enc_weights_v"+self.params.version+"-"
      +current_step.zfill(5)+".png"))

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    feed_dict = super(VAE, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
    if(is_test): # Turn off dropout when not training
      feed_dict[self.dropout_keep_probs] = [1.0,] * len(self.params.dropout)
    else:
      feed_dict[self.dropout_keep_probs] = self.params.dropout
    return feed_dict


class params(BaseParams):
  def __init__(self):
    super(params, self).__init__()
    self.model_type = "vae_test"
    self.model_name = "vae_test"
    self.version = "0.0"
    self.data_type = "mnist"
    self.vectorize_data = False
    self.rescale_data = True
    self.batch_size = 64
    self.num_samples = 1000
    self.input_max = 100
    self.output_channels = [64, 64, 64, 8]
    self.optimizer = "adam"#"annealed_sgd"
    self.activation_functions = ["lrelu", "lrelu", "lrelu", "lrelu", "lrelu", "relu", "relu", "relu", "sigmoid"]
    self.dropout = [0.8]*5

    self.log_int = 100
    self.cp_int = 1e4
    self.gen_plot_int = 1e4
    self.schedule = [
      {"num_batches": int(3e4),
       "weights": None,
       "weight_lr": 5e-4,
       "decay_steps": int(3e4*0.8),
       "decay_rate": 0.5,
       "staircase": True}]

vae_params = params()
vae_model = VAE()

data = ds.get_data(vae_params)
data = vae_model.preprocess_dataset(data, vae_params)
data = vae_model.reshape_dataset(data, vae_params)
vae_params.data_shape = list(data["train"].shape[1:])

vae_model.setup(vae_params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config, graph=vae_model.graph) as sess:
  sess.run(vae_model.init_op)
  sess.graph.finalize()
  init = True
  for sch_idx, sch in enumerate(vae_params.schedule):
    for b_step in np.arange(sch["num_batches"]):
      data_batch = data["train"].next_batch(vae_params.batch_size)
      input_data = data_batch[0]
      input_labels = data_batch[1]

      feed_dict = vae_model.get_feed_dict(input_data, None)
      feed_dict[vae_model.Y] = input_data

      if init:
        vae_model.print_update(input_data)
        init=False

      sess_run_list = []
      for w_idx in range(len(vae_model.get_schedule("weights"))):
        sess_run_list.append(vae_model.apply_grads[sch_idx][w_idx])
      sess.run(sess_run_list, feed_dict)

      current_step = sess.run(vae_model.global_step)
      if (current_step % vae_params.log_int == 0):
        vae_model.print_update(input_data, batch_step=b_step+1)
      if (current_step % vae_params.gen_plot_int == 0):
        vae_model.generate_plots(input_data)

  vae_model.print_update(input_data, batch_step=b_step+1)
  save_dir = vae_model.write_checkpoint(sess)

  # Generate images
  randoms = [np.random.normal(0, 1, vae_model.n_latent) for _ in range(10)]
  feed_dict = {vae_model.sampled: randoms}
  feed_dict[vae_model.dropout_keep_probs] = [1.0] * len(vae_model.params.dropout)
  imgs = sess.run(vae_model.dec, feed_dict)
  imgs = np.stack([np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))], axis=0)
  imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
  gen_imgs_fig = pf.plot_data_tiled(imgs[..., None], normalize=False,
    title="Generated images", vmin=0, vmax=1,
    save_filename=(vae_model.params.disp_dir+"generated_images"
    +"_v"+vae_model.params.version+"_"+str(current_step).zfill(5)+".png"))

log_file = vae_model.logger.filename
logger = Logger(filename=None)
log_text = logger.load_file(log_file)
model_stats = logger.read_stats(log_text)
keys = [
  "recon_loss",
  "latent_loss",
  "total_loss"]
labels = [
  "Latent Loss",
  "Recon Loss",
  "Total Loss"]
stats_fig = pf.plot_stats(model_stats, keys=keys, labels=labels, figsize=(8,8),
  save_filename=vae_model.params.disp_dir+vae_model.params.model_name+"_train_stats.png")
