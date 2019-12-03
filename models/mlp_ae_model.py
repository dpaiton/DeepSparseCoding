import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from models.ae_model import AeModel
from models.mlp_model import MlpModel
from modules.mlp_module import MlpModule
from modules.activations import activation_picker

class MlpAeModel(MlpModel):
  def load_params(self, params):
    """
    Load parameters into object
    Inputs:
     params: [obj] model parameters
    """
    super(MlpAeModel, self).load_params(params)
    AeModel.ae_load_params(self, params)
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

  def build_ae_module(self, input_node):
    module = AeModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.decay_mult, self.norm_mult, self.ae_act_funcs, self.ae_dropout_keep_probs,
      self.params.tie_dec_weights, self.params.norm_w_init, variable_scope="ae")
    return module

  #def build_mlp_module(self, input_node):
  #  module = MlpModule(input_node, self.label_placeholder, self.params.mlp_layer_types,
  #    self.params.mlp_output_channels, self.params.batch_norm, self.mlp_dropout_keep_probs,
  #    self.params.max_pool, self.params.max_pool_ksize, self.params.max_pool_strides,
  #    self.params.mlp_patch_size, self.params.mlp_conv_strides, self.mlp_act_funcs,
  #    self.params.eps, lrn=self.params.lrn, loss_type="softmax_cross_entropy",
  #    decay_mult=self.params.mlp_decay_mult, norm_mult=self.params.mlp_norm_mult)
  #  return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.label_placeholder = tf.compat.v1.placeholder(tf.float32,
            shape=self.label_shape, name="input_labels")
          self.decay_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="decay_mult")
          self.norm_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="norm_mult")
          self.train_ae = tf.compat.v1.placeholder(tf.bool, shape=(), name="train_ae")

        with tf.compat.v1.variable_scope("placeholders") as sess:
          self.mlp_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="mlp_dropout_keep_probs")
          self.ae_dropout_keep_probs = tf.compat.v1.placeholder(tf.float32, shape=[None],
            name="ae_dropout_keep_probs")

        self.train_ae = tf.cast(self.train_ae, tf.float32)

        self.ae_module = self.build_ae_module(input_node)
        self.trainable_variables.update(self.ae_module.trainable_variables)

        if self.params.train_on_recon:
          if self.params.mlp_layer_types[0] == "conv":
            data_shape = [tf.shape(input_node)[0]]+self.params.full_data_shape
            mlp_input = tf.reshape(self.ae_module.reconstruction, shape=data_shape)
          elif self.params.mlp_layer_types[0] == "fc":
            mlp_input = self.ae_module.reconstruction
          else:
            assert False, ("params.mlp_layer_types must be 'fc' or 'conv'")
        else: # train on VAE latent encoding
          assert self.params.mlp_layer_types[0] == "fc", (
            "MLP must have FC layers to train on SAE activity")
          mlp_input = self.ae_module.a
        self.mlp_module = self.build_mlp_module(mlp_input)
        self.trainable_variables.update(self.mlp_module.trainable_variables)

        with tf.compat.v1.variable_scope("loss") as scope:
          #Loss switches based on train_ae flag
          self.total_loss = self.train_ae * self.ae_module.total_loss + \
            (1-self.train_ae) * self.mlp_module.total_loss

        self.label_est = tf.identity(self.mlp_module.label_est, name="label_est")

        with tf.compat.v1.variable_scope("performance_metrics") as scope:
          #VAE metrics
          MSE = tf.reduce_mean(tf.square(tf.subtract(input_node, self.ae_module.reconstruction)),
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
    feed_dict = super(MlpAeModel, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
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
    update_dict = super(MlpAeModel, self).generate_update_dict(input_data, input_labels,
      batch_step)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.ae_module.loss_dict["recon_loss"],
      self.ae_module.a, self.ae_module.reconstruction,
      self.pSNRdB]

    out_vals =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    recon_loss, ae_a_vals, recon, pSNRdB\
      = out_vals[0:5]

    input_max = np.max(input_data)
    input_mean = np.mean(input_data)
    input_min = np.min(input_data)
    recon_max = np.max(recon)
    recon_mean = np.mean(recon)
    recon_min = np.min(recon)
    ae_a_vals_max = np.array(ae_a_vals.max())
    ae_a_vals_mean = np.array(ae_a_vals.mean())
    ae_a_vals_min = np.array(ae_a_vals.min())
    ae_a_frac_act = np.array(np.count_nonzero(ae_a_vals)
      / float(ae_a_vals.size))

    stat_dict = {
      "ae_recon_loss":recon_loss,
      "pSNRdB": np.mean(pSNRdB),
      "ae_a_fraction_active":ae_a_frac_act,
      "ae_a_max_mean_min":[ae_a_vals_max, ae_a_vals_mean, ae_a_vals_min],
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
    super(MlpAeModel, self).generate_plots(input_data, input_labels)

    feed_dict = self.get_feed_dict(input_data, input_labels)

    eval_list = [self.global_step, self.ae_module.w_list[0],
      self.ae_module.reconstruction, self.ae_module.a]
    eval_out = tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    weights, recon, ae_activity = eval_out[1:]
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
    fig = pf.plot_activity_hist(ae_activity, title="AE Activity Histogram",
      save_filename=self.params.disp_dir+"ae_act_hist"+filename_suffix)
    fig = pf.plot_data_tiled(weights, normalize=False,
      title="Dictionary at step "+current_step, vmin=None, vmax=None,
      save_filename=self.params.disp_dir+"phi"+filename_suffix)
