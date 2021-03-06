import tensorflow as tf

from DeepSparseCoding.tf1x.utils.trainable_variable_dict import TrainableVariableDict
import DeepSparseCoding.tf1x.utils.entropy_functions as ef
from DeepSparseCoding.tf1x.modules.ae_module import AeModule
from DeepSparseCoding.tf1x.modules.activations import sigmoid

class SaeModule(AeModule):
  def __init__(self, data_tensor, layer_types, enc_channels, dec_channels, patch_size,
    conv_strides, sparse_mult, w_decay_mult, w_norm_mult, target_act, act_funcs, dropout,
    tie_dec_weights, w_init_type, variable_scope="sae"):
    """
    Implementation of sparse autoencoder described in Andrew Ng's 2011 Stanford CS294A lecture notes
    Sigmoidal activation function
    Untied encoding & decoding weights
    Linear reconstructions - input images do not have 0-1 range
    Inputs:
      data_tensor
      enc_channels [list of ints] the number of output channels per encoder layer
        Last entry is the number of latent units
      dec_channels [list of ints] the number of output channels per decoder layer
        Last entry must be the number of input pixels for FC layers and channels for CONV layers
      w_decay_mult -  weight decay multiplier
      w_norm_mult: tradeoff multiplier for weight norm loss (asks weight norm to == 1)
      act_funcs - activation functions
      dropout: specifies the keep probability or None
      conv: if True, do convolution
      conv_strides: list of strides for convolution [batch, y, x, channels]
      patch_size: number of (y, x) inputs for convolutional patches
      w_init_type: if True, l2 normalize w_init,
        reducing over [0] axis on enc and [-1] axis on dec
      variable_scope - specifies the variable_scope for the module
    Outputs:
      dictionary
    """
    self.sparse_mult = sparse_mult
    self.target_act = target_act
    super(SaeModule, self).__init__(data_tensor, layer_types, enc_channels, dec_channels,
      patch_size, conv_strides, w_decay_mult, w_norm_mult, act_funcs, dropout, tie_dec_weights,
      w_init_type, variable_scope)

  def compute_sparse_loss(self, a_in):
    with tf.compat.v1.variable_scope("unsupervised"):
      reduc_dims = tuple(range(len(a_in.get_shape().as_list()) - 1))
      avg_act = tf.reduce_mean(input_tensor=a_in, axis=reduc_dims, name="batch_avg_activity")
      p_dist = self.target_act * tf.subtract(ef.safe_log(self.target_act),
        ef.safe_log(avg_act), name="kl_p")
      q_dist = (1-self.target_act) * tf.subtract(ef.safe_log(1-self.target_act),
        ef.safe_log(1-avg_act), name="kl_q")
      kl_divergence = tf.reduce_sum(input_tensor=tf.add(p_dist, q_dist), name="kld")
      sparse_loss = tf.multiply(self.sparse_mult, kl_divergence, name="sparse_loss")
    return sparse_loss

  def compute_total_loss(self):
    with tf.compat.v1.variable_scope("loss") as scope:
      self.loss_dict = {"recon_loss":self.compute_recon_loss(self.reconstruction),
        "sparse_loss":self.compute_sparse_loss(self.a),
        "weight_decay_loss":self.compute_weight_decay_loss(),
        "weight_norm_loss":self.compute_weight_norm_loss()}
      self.total_loss = tf.add_n([loss for loss in self.loss_dict.values()], name="total_loss")
