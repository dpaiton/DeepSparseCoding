import numpy as np
import tensorflow as tf
#import tensorflow_compression as tfc
import utils.plot_functions as pf
import utils.data_processing as dp
from models.base_model import Model

class Johannes_GDN(Model):
  """
  Density modeling of images using a generalized normalization transformation
  J. Ballé, V. Laparra, E.P. Simoncelli
  https://arxiv.org/abs/1511.06281

  Code adapted from:
    https://github.com/tensorflow/compression/blob/master/examples/bls2017.py
    https://tensorflow.github.io/compression/docs/entropy_bottleneck.html
    https://groups.google.com/forum/#!forum/tensorflow-compression
  """
  def __init__(self):
    super(Johannes_GDN, self).__init__()
    self.vector_inputs = False

  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [dict] model parameters
    Modifiable Parameters:
      num_filters: num filters for all layers
    """
    super(Johannes_GDN, self).load_params(params)
    self.num_filters = params["num_filters"]
    self.num_pixels = params["num_pixels"]

  def analysis_transform(tensor, num_filters):
    """Builds the analysis transform."""

    with tf.variable_scope("analysis"):
      with tf.variable_scope("layer_0"):
        layer = tfc.SignalConv2D(
            num_filters, (9, 9), corr=True, strides_down=4, padding="same_zeros",
            use_bias=True, activation=tfc.GDN())
        tensor = layer(tensor)

      with tf.variable_scope("layer_1"):
        layer = tfc.SignalConv2D(
            num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN())
        tensor = layer(tensor)

      with tf.variable_scope("layer_2"):
        layer = tfc.SignalConv2D(
            num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
            use_bias=False, activation=None)
        tensor = layer(tensor)

      return tensor

  def synthesis_transform(tensor, num_filters):
    """Builds the synthesis transform."""
    with tf.variable_scope("synthesis"):
      with tf.variable_scope("layer_0"):
        layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
        tensor = layer(tensor)

      with tf.variable_scope("layer_1"):
        layer = tfc.SignalConv2D(
          num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
          use_bias=True, activation=tfc.GDN(inverse=True))
        tensor = layer(tensor)

      with tf.variable_scope("layer_2"):
        layer = tfc.SignalConv2D(
          3, (9, 9), corr=False, strides_up=4, padding="same_zeros",
          use_bias=True, activation=None)
        tensor = layer(tensor)

      return tensor

  def build_graph(self):
    """Build the TensorFlow graph object"""
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.x = tf.placeholder(tf.float32, shape=self.x_shape, name="input_data")
          self.ent_mult = tf.placeholder(tf.float32, shape=(), name="ent_mult")

        with tf.name_scope("inference") as scope:
          self.a = analysis_transform(self.x, args.num_filters)
          self.entropy_bottleneck = tfc.EntropyBottleneck()
          y_tilde, likelihoods = self.entropy_bottleneck(self.a, training=True)
          self.x_ = synthesis_transform(y_tilde, args.num_filters)

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.w_list = tf.trainable_variables()
        # Total number of bits divided by number of pixels
        self.train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * self.num_pixels)
        # Mean squared error across pixels
        self.train_mse = tf.reduce_sum(tf.squared_difference(x, x_tilde)) / self.num_pixels
        # Multiply by 255^2 to correct for rescaling
        #train_mse *= 255 ** 2
        # The rate-distortion cost
        self.total_loss = self.train_mse + self.ent_mult * self.train_bpp

  def add_optimizers_to_graph(self):
    """
    Add optimizers to graph
    Creates member variables grads_and_vars and apply_grads for each weight
      - both member variables are indexed by [schedule_idx][weight_idx]
      - grads_and_vars holds the gradients for the weight updates
      - apply_grads is the operator to be called to perform weight updates
    """
    with tf.device(self.device):
      with self.graph.as_default():
        main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        main_step = main_optimizer.minimize(self.total_loss, global_step=self.global_step)
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(self.entropy_bottleneck.losses[0])
        self.apply_grads = tf.group(main_step, aux_step, self.entropy_bottleneck.updates[0])
    self.optimizers_added = True

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Generates a dictionary to be logged in the print_update function
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.w_list, self.a, self.x_, self.total_loss, self.train_mse]
    #loss_list = [self.loss_dict[key] for key in self.loss_dict.keys()]
    #eval_list = [self.global_step]+loss_list+[self.total_loss, self.a, self.u_list[-1],
    #  self.batch_MSE, self.SNRdB]
    #init_eval_length = len(eval_list)
    #grad_name_list = []
    #learning_rate_dict = {}
    #for w_idx, weight_grad_var in enumerate(self.grads_and_vars[self.sched_idx]):
    #  eval_list.append(weight_grad_var[0][0]) # [grad(0) or var(1)][value(0) or name[1]]
    #  grad_name = weight_grad_var[0][1].name.split('/')[1].split(':')[0] #2nd is np.split
    #  grad_name_list.append(grad_name)
    #  learning_rate_dict[grad_name] = self.get_schedule("weight_lr")[w_idx]
    out_vals =  tf.get_default_session().run(eval_list, feed_dict)
    current_step = out_vals[0]
    #losses = out_vals[1:len(loss_list)+1]
    #total_loss, a_vals, recon, MSE, SNRdB = out_vals[len(loss_list)+1:init_eval_length]
    w_list, a_vals, recon, total_loss, MSE = out_vals[1:]
    input_mean = np.mean(input_data)
    input_max = np.max(input_data)
    input_min = np.min(input_data)
    recon_mean = np.mean(recon)
    recon_max = np.max(recon)
    recon_min = np.min(recon)
    a_vals_max = np.array(a_vals.max())
    a_vals_min = np.array(a_vals.min())
    a_frac_act = np.array(np.count_nonzero(a_vals)
      / float(a_vals.size))
    MSE = np.array(MSE)
    stat_dict = {"global_batch_index":current_step,
      "batch_step":batch_step,
      "schedule_index":self.sched_idx,
      "total_loss":total_loss,
      "a_max":a_vals_max,
      "a_min":a_vals_min,
      "a_fraction_active":a_frac_act,
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min],
      "MSE":MSE}
    #for idx, key in enumerate(self.loss_dict.keys()):
    #  stat_dict[key] = losses[idx]
    #grads = out_vals[init_eval_length:]
    #for grad, name in zip(grads, grad_name_list):
    #  stat_dict[name+"_max_grad"] = learning_rate_dict[name]*np.array(grad.max())
    #  stat_dict[name+"_min_grad"] = learning_rate_dict[name]*np.array(grad.min())
    return stat_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.a, self.x_, self.w_list]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    assert np.all(np.stack([np.all(np.isfinite(arry)) for arry in eval_out])), (
      "Some plot evals had non-finite values")
    current_step = str(eval_out[0])
    start = 1
    a_vals, recon, w_list = eval_out[1:]
    #w_enc_shape = w_enc.shape
    #w_enc_norm = np.linalg.norm(w_enc.reshape([np.prod(w_enc_shape[:-1]), w_enc_shape[-1]]),
    #  axis=1, keepdims=False)
    #w_dec = np.transpose(w_dec, axes=(0,1,3,2))
    #w_dec_shape = w_dec.shape
    #w_dec_norm = np.linalg.norm(w_dec.reshape([np.prod(w_dec_shape[:-1]), w_dec_shape[-1]]),
    #  axis=1, keepdims=False)
    #w_enc = np.transpose(w_enc, axes=(3,0,1,2))
    #w_enc = dp.reshape_data(w_enc, flatten=True)[0]
    #fig = pf.plot_data_tiled(w_enc, normalize=False,
    #  title="Encoding weights at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"w_enc_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #w_dec = np.transpose(w_dec, axes=(2,0,1,3))
    #w_dec = dp.reshape_data(w_dec, flatten=True)[0]
    #fig = pf.plot_data_tiled(w_dec, normalize=False,
    #  title="Decoding weights at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"w_dec_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #fig = pf.plot_bar(w_enc_norm, num_xticks=5,
    #  title="w_enc l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
    #  save_filename=(self.disp_dir+"w_enc_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    #fig = pf.plot_bar(w_dec_norm, num_xticks=5,
    #  title="w_dec l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
    #  save_filename=(self.disp_dir+"w_dec_norm_v"+self.version+"-"+current_step.zfill(5)+".png"))
    #for idx, w_gdn in enumerate(w_gdn_eval_list):
    #  fig = pf.plot_weight_image(w_gdn, title="GDN "+str(idx)+" Weights", figsize=(10,10),
    #    save_filename=(self.disp_dir+"w_gdn_"+str(idx)+"_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    #for idx, b_gdn in enumerate(b_gdn_eval_list):
    #  fig = pf.plot_activity_hist(b_gdn, title="GDN "+str(idx)+" Bias Histogram",
    #    save_filename=(self.disp_dir+"b_gdn_"+str(idx)+"_hist_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    #for idx, bias in enumerate(b_eval_list):
    #  fig = pf.plot_activity_hist(bias, title="Bias "+str(idx)+" Histogram",
    #    save_filename=(self.disp_dir+"b_"+str(idx)+"_hist_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    #for idx, gdn_mult_eval in enumerate(gdn_mult_eval_list):
    #  gdn_mult_resh = gdn_mult_eval.reshape(np.prod(gdn_mult_eval.shape))
    #  fig = pf.plot_activity_hist(gdn_mult_resh, title="GDN Multiplier Histogram",
    #    save_filename=(self.disp_dir+"gdn_mult_v"+self.version+"-"
    #    +current_step.zfill(5)+".png"))
    fig = pf.plot_activity_hist(a_vals, title="Activity Histogram (pre-mem)",
      save_filename=(self.disp_dir+"act_hist_v"+self.version+"-"
      +current_step.zfill(5)+".png"))
    #input_data, input_orig_shape = dp.reshape_data(input_data, flatten=True)[:2]
    #fig = pf.plot_activity_hist(input_data, title="Image Histogram",
    #  save_filename=(self.disp_dir+"img_hist_"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    #input_data = dp.reshape_data(input_data, flatten=False, out_shape=input_orig_shape)[0]
    #fig = pf.plot_data_tiled(input_data, normalize=False,
    #  title="Images at step "+current_step, vmin=None, vmax=None,
    #  save_filename=(self.disp_dir+"images_"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    recon = dp.reshape_data(recon, flatten=False)[0]
    fig = pf.plot_data_tiled(recon, normalize=False,
      title="Recons at step "+current_step, vmin=None, vmax=None,
      save_filename=(self.disp_dir+"recons_v"+self.version+"-"+current_step.zfill(5)+".png"))
