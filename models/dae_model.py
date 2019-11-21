import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.ae_model import AeModel
from modules.dae_module import DaeModule

class DaeModel(AeModel):
  def __init__(self):
    """
    Divisive Autoencoder
    """
    super(DaeModel, self).__init__()

  def build_module(self, input_node):
    module = DaeModule(input_node, self.params.layer_types, self.params.output_channels,
      self.params.patch_size, self.params.conv_strides, self.ent_mult, self.decay_mult,
      self.norm_mult, self.params.bounds_slope, self.params.latent_min, self.params.latent_max,
      self.params.num_triangles, self.params.mle_step_size, self.params.num_mle_steps,
      self.params.num_quant_bins, self.noise_var_mult, self.params.gdn_w_init_const,
      self.params.gdn_b_init_const, self.params.gdn_w_thresh_min, self.params.gdn_b_thresh_min,
      self.params.gdn_eps, self.act_funcs, self.dropout_keep_probs,
      self.params.tie_decoder_weights, self.params.norm_w_init, variable_scope="dae")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.ent_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="entropy_mult")
          self.noise_var_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="noise_variance_mult")

    super(DaeModel, self).build_graph_from_input(input_node)

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("probability_estimate") as scope:
          self.mle_update = self.module.mle_update

  def get_encodings(self):
    return self.a

  def get_total_loss(self):
    return self.module.total_loss

  def generate_update_dict(self, input_data, input_labels=None, batch_step=0):
    """
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
      batch_step: current batch number within the schedule
    """
    update_dict = super(DaeModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.module.loss_dict["entropy_loss"], self.module.loss_dict["ramp_loss"]]
    eval_out =  tf.compat.v1.get_default_session().run(eval_list, feed_dict)
    stat_dict = {"entropy_loss": eval_out[0],
      "ramp_loss": eval_out[1]}
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(DaeModel, self).generate_plots(input_data, input_labels)
