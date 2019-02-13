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
    module = SaeModule(input_node, self.params.output_channels, self.sparse_mult, self.decay_mult,
      self.target_act, self.act_funcs, self.dropout_keep_probs, self.params.tie_decoder_weights,
      self.params.conv, self.conv_strides, self.patch_y, self.patch_x, name_scope="SAE")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.target_act = tf.placeholder(tf.float32, shape=(), name="target_act")
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
    sparse_loss =  tf.get_default_session().run(self.module.loss_dict["sparse_loss"], feed_dict)
    stat_dict = {"sparse_loss":sparse_loss}
    update_dict.update(stat_dict)
    return update_dict
