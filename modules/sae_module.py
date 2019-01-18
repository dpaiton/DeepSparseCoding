import numpy as np
import tensorflow as tf
from utils.trainable_variable_dict import TrainableVariableDict
import utils.entropy_functions as ef
from modules.ae_module import AeModule
from modules.activations import sigmoid

class SaeModule(AeModule):
  def __init__(self, data_tensor, output_channels, sparse_mult, decay_mult, target_act,
    act_funcs, dropout, name="SAE"):
    """
    Implementation of sparse autoencoder described in Andrew Ng's 2011 Stanford CS294A lecture notes
    Sigmoidal activation function
    Untied encoding & decoding weights
    Linear reconstructions - input images do not have 0-1 range
    Inputs:
      data_tensor
      output_channels - a list of channels to make, also defines number of layers
      decay_mult - weight decay multiplier
      act_funcs - activation functions
      dropout - specifies the keep probability or None
      name
    Outputs:
      dictionary
    """
    self.sparse_mult = sparse_mult
    self.target_act = target_act
    super(SaeModule, self).__init__(data_tensor, output_channels, decay_mult, act_funcs,
      dropout, name)

  def compute_sparse_loss(self, a_in):
    with tf.name_scope("unsupervised"):
      avg_act = tf.reduce_mean(a_in, axis=[0], name="batch_avg_activity")
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

  def build_graph(self):
    with tf.name_scope("weight_inits") as scope:
      self.w_init = tf.initializers.truncated_normal(mean=0.0, stddev=0.01, dtype=tf.float32)
      self.b_init = tf.initializers.constant(1e-4, dtype=tf.float32)

    self.u_list = [self.data_tensor]
    self.w_list = []
    self.b_list = []
    enc_u_list, enc_w_list, enc_b_list = self.build_encoder(self.u_list[0],
      self.act_funcs[:self.num_encoder_layers], self.w_shapes[:self.num_encoder_layers])
    self.u_list += enc_u_list
    self.w_list += enc_w_list
    self.b_list += enc_b_list

    with tf.variable_scope("inference") as scope:
      self.a = tf.identity(enc_u_list[-1], name="activations")

    dec_u_list, dec_w_list, dec_b_list = self.build_decoder(self.num_encoder_layers,
      enc_u_list[-1], self.act_funcs[self.num_encoder_layers:],
      self.w_shapes[self.num_encoder_layers:])
    self.u_list += dec_u_list
    self.w_list += dec_w_list
    self.b_list += dec_b_list

    for w,b in zip(self.w_list, self.b_list):
      self.trainable_variables[w.name] = w
      self.trainable_variables[b.name] = b

    with tf.name_scope("output") as scope:
      self.reconstruction = tf.identity(self.u_list[-1], name="reconstruction")

    with tf.name_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "sparse_loss":self.compute_sparse_loss(self.a),
        "weight_decay_loss":self.compute_weight_decay_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
