import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
from models.ae_model import AeModel
from modules.sae_module import SaeModule

class SaeModel(AeModel):
  def __init__(self):
    super(SaeModel, self).__init__()

  def build_module(self, input_node):
    module = SaeModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.sparse_mult, self.w_decay_mult, self.w_norm_mult, self.target_act, self.act_funcs,
      self.ae_dropout_keep_probs, self.params.tie_dec_weights, self.params.w_init_type,
      variable_scope="sae")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.sparse_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.target_act = tf.compat.v1.placeholder(tf.float32, shape=(), name="target_act")
    super(SaeModel, self).build_graph_from_input(input_node)

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
    update_dict = super(SaeModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    sparse_loss =  tf.compat.v1.get_default_session().run(self.module.loss_dict["sparse_loss"], feed_dict)
    stat_dict = {"sparse_loss":sparse_loss}
    update_dict.update(stat_dict)
    return update_dict
