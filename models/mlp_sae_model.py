import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from modules.sae_module import SaeModule
from models.sae_model import SaeModel
from models.mlp_model import MlpModel
from modules.mlp_module import MlpModule
from modules.activations import activation_picker

class MlpSaeModel(MlpModel):
  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpSaeModel, self).load_params(params)
    SaeModel.ae_load_params(self, params)
    # Network Size
    self.vector_inputs = True
    self.input_shape = [None, self.params.num_pixels]
    self.label_shape = [None, self.params.num_classes]
    self.ae_act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.ae_activation_functions]
    self.mlp_act_funcs = [activation_picker(act_func_str)
      for act_func_str in self.params.mlp_activation_functions]
    if np.all([layer_type == "fc" for layer_type in self.params.ae_layer_types]):
      self.params.ae_patch_size = []
      self.params.ae_conv_strides = []
    if np.all([layer_type == "fc" for layer_type in self.params.mlp_layer_types]):
      self.params.mlp_patch_size = []
      self.params.mlp_conv_strides = []

  def build_sae_module(self, input_node):
    module = SaeModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.sparse_mult, self.w_decay_mult, self.w_norm_mult, self.target_act, self.ae_act_funcs,
      self.ae_dropout_keep_probs, self.params.tie_dec_weights, self.params.w_init_type,
      variable_scope="sae")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.label_placeholder = tf.compat.v1.placeholder(tf.float32,
            shape=self.label_shape, name="input_labels")
          self.w_decay_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="w_decay_mult")
          self.w_norm_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="w_norm_mult")
          self.sparse_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.target_act = tf.compat.v1.placeholder(tf.float32, shape=(), name="target_act")
          self.train_sae = tf.compat.v1.placeholder(tf.bool, shape=(), name="train_sae")

        with tf.compat.v1.variable_scope("placeholders") as sess:
          self.mlp_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="mlp_dropout_keep_probs")
          self.ae_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="ae_dropout_keep_probs")

        self.train_sae = tf.cast(self.train_sae, tf.float32)

        self.sae_module = self.build_sae_module(input_node)
        self.trainable_variables.update(self.sae_module.trainable_variables)

        if self.params.train_on_recon:
          if self.params.mlp_layer_types[0] == "conv":
            data_shape = [tf.shape(input_node)[0]]+self.params.full_data_shape
            mlp_input = tf.reshape(self.sae_module.reconstruction, shape=data_shape)
          elif self.params.mlp_layer_types[0] == "fc":
            mlp_input = self.sae_module.reconstruction
          else:
            assert False, ("params.mlp_layer_types must be 'fc' or 'conv'")
        else: # train on VAE latent encoding
          assert self.params.mlp_layer_types[0] == "fc", (
            "MLP must have FC layers to train on SAE activity")
          mlp_input = self.sae_module.a
        self.mlp_module = self.build_mlp_module(mlp_input)
        self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.compat.v1.variable_scope("loss") as scope:
          #Loss switches based on train_sae flag
          self.total_loss = self.train_sae * self.sae_module.total_loss + \
            (1-self.train_sae) * self.mlp_module.total_loss

        self.label_est = tf.identity(self.mlp_module.label_est, name="label_est")

        with tf.compat.v1.variable_scope("performance_metrics") as scope:
          #VAE metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.sae_module.reconstruction)),
            axis=[1, 0], name="mean_squared_error")
          pixel_var = tf.nn.moments(input_node, axes=[1])[1]
          self.pSNRdB = tf.multiply(10.0, ef.safe_log(tf.math.divide(tf.square(pixel_var), MSE)),
            name="recon_quality")
          with tf.compat.v1.variable_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.label_est, axis=1),
              tf.argmax(self.label_placeholder, axis=1), name="individual_accuracy")
          with tf.compat.v1.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    feed_dict = super(MlpSaeModel, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
    if(is_test): # Turn off dropout when not training
      feed_dict[self.ae_dropout_keep_probs] = [1.0,] * len(self.params.ae_dropout)
    else:
      feed_dict[self.ae_dropout_keep_probs] = self.params.ae_dropout
    return feed_dict

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
    update_dict = super(MlpSaeModel, self).generate_update_dict(input_data, input_labels,
      batch_step)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.sae_module.loss_dict["recon_loss"],
      self.sae_module.loss_dict["sparse_loss"],
      self.sae_module.a, self.sae_module.reconstruction,
      self.pSNRdB]

    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    recon_loss, sparse_loss, sae_a_vals, recon, pSNRdB\
      = out_vals[0:5]

    input_max = np.max(input_data)
    input_mean = np.mean(input_data)
    input_min = np.min(input_data)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    sae_a_vals_max = np.array(sae_a_vals.max())
    sae_a_vals_mean = np.array(sae_a_vals.mean())
    sae_a_vals_min = np.array(sae_a_vals.min())
    sae_a_frac_act = np.array(np.count_nonzero(sae_a_vals)
      / float(sae_a_vals.size))

    stat_dict = {
      "sae_recon_loss":recon_loss,
      "sae_sparse_loss":sparse_loss,
      "pSNRdB": np.mean(pSNRdB),
      "sae_a_fraction_active":sae_a_frac_act,
      "sae_a_max_mean_min":[sae_a_vals_max, sae_a_vals_mean, sae_a_vals_min],
      "x_max_mean_min":[input_max, input_mean, input_min],
      "x_hat_max_mean_min":[recon_max, recon_mean, recon_min]}
    update_dict.update(stat_dict) #stat_dict overwrites for same keys
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(MlpSaeModel, self).generate_plots(input_data, input_labels)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.sae_module.w_list[0],
      self.sae_module.reconstruction, self.sae_module.a]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    weights, recon, sae_activity = eval_out[1:]
    fig = pf.plot_activity_hist(input_data, title="Image Histogram",
      save_filename=self.params.disp_dir+"img_hist"+filename_suffix)
    fig = pf.plot_activity_hist(recon, title="Recon Histogram",
      save_filename=self.params.disp_dir+"recon_hist"+filename_suffix)
    weights_norm = np.linalg.norm(weights, axis=0, keepdims=False)
    recon = dp.reshape_data(recon, flatten=False)[0]
    weights = dp.reshape_data(weights.T, flatten=False)[0] # [num_neurons, height, width]
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
    fig = pf.plot_activity_hist(sae_activity, title="SAE Activity Histogram",
      save_filename=self.params.disp_dir+"sae_act_hist"+filename_suffix)
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)
