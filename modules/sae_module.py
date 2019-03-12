import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from modules.activations import sigmoid

class SaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, output_channels, patch_size, conv_strides,
    sparse_mult, decay_mult, target_act, act_funcs, dropout, tie_decoder_weights,
    variable_scope="sae"):
    """
    Implementation of sparse autoencoder described in Andrew Ng's 2011 Stanford CS294A lecture notes
    Sigmoidal activation function
    Untied encoding & decoding weights
    Linear reconstructions - input images do not have 0-1 range
    Inputs:
      data_tensor
      output_channels - a list of channels to make, also defines number of layers
      decay_mult -  weight decay multiplier
      act_funcs - activation functions
      dropout: specifies the keep probability or None
      conv: if True, do convolution
      conv_strides: list of strides for convolution [batch, y, x, channels]
      patch_y: number of y inputs for convolutional patches
      patch_x: number of x inputs for convolutional patches
      variable_scope - specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.sparse_mult = sparse_mult
    self.target_act = target_act
    super(SaeModule, self).__init__(data_tensor, layer_types, output_channels,
      patch_size, conv_strides, decay_mult, act_funcs, dropout,
      tie_decoder_weights, variable_scope)

  def compute_sparse_loss(self, a_in):
    with tf.variable_scope("unsupervised"):
      reduc_dims = tuple(range(len(a_in.get_shape().as_list()) - 1))
      avg_act = tf.reduce_mean(a_in, axis=reduc_dims, name="batch_avg_activity")
      p_dist = self.target_act * tf.subtract(ef.safe_log(self.target_act),
        ef.safe_log(avg_act), name="kl_p")
      q_dist = (1-self.target_act) * tf.subtract(ef.safe_log(1-self.target_act),
        ef.safe_log(1-avg_act), name="kl_q")
      #p_dist = self.target_act * tf.subtract(tf.log(self.target_act), tf.log(avg_act), name="kl_p")
      #q_dist = (1-self.target_act) * tf.subtract(tf.log(1-self.target_act), tf.log(1-avg_act),
      #  name="kl_q")
      kl_divergence = tf.reduce_sum(tf.add(p_dist, q_dist), name="kld")
      sparse_loss = tf.multiply(self.sparse_mult, kl_divergence, name="sparse_loss")
    return sparse_loss

  def compute_total_loss(self):
    with tf.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "sparse_loss":self.compute_sparse_loss(self.a),
        "weight_decay_loss":self.compute_weight_decay_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
