import numpy as np
import tensorflow as tf

import DeepSparseCoding.tf1x.utils.plot_functions as pf
import DeepSparseCoding.tf1x.utils.data_processing as dp
import DeepSparseCoding.tf1x.utils.entropy_functions as ef
from DeepSparseCoding.tf1x.models.dae_model import DaeModel
from DeepSparseCoding.tf1x.modules.dae_mem_module import DaeMemModule

class DaeMemModel(DaeModel):
  def __init__(self):
    """
    Divisive Memristor Autoencoder
    """
    super(DaeMemModel, self).__init__()

  def build_module(self, input_node):
    module = DaeMemModule(input_node, self.params.ae_layer_types, self.params.ae_enc_channels,
      self.params.ae_dec_channels, self.params.ae_patch_size, self.params.ae_conv_strides,
      self.ent_mult, self.w_decay_mult, self.w_norm_mult, self.params.bounds_slope,
      self.params.latent_min, self.params.latent_max, self.params.num_triangles,
      self.params.mle_step_size, self.params.num_mle_steps, self.params.gdn_w_init_const,
      self.params.gdn_b_init_const, self.params.gdn_w_thresh_min, self.params.gdn_b_thresh_min,
      self.params.gdn_eps, self.params.memristor_data_loc, self.params.memristor_type,
      self.memristor_std_eps, self.params.synthetic_noise, self.params.mem_error_rate,
      self.act_funcs, self.ae_dropout_keep_probs, self.params.tie_dec_weights,
      self.params.w_init_type, variable_scope="dae_mem")
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object
    This is exactly the same as the parent call but just removes noise_var_mult placeholder
    """
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("auto_placeholders") as scope:
          self.ent_mult = tf.compat.v1.placeholder(tf.float32, shape=(), name="entropy_mult")

        # this is where you construct mem noise using self.num_latent#
        with tf.compat.v1.variable_scope("placeholders") as scope:
          self.memristor_std_eps = tf.compat.v1.placeholder(tf.float32, shape=[None, None],
            name="memristor_std_eps")
        # REMEMBER!!!: If using memristor_type = 'rram', std_eps will be the width of the uniform noise, not gauss mult
    super(DaeModel, self).build_graph_from_input(input_node)

    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.compat.v1.variable_scope("probability_estimate") as scope:
          self.mle_update = self.module.mle_update

  def get_feed_dict(self, input_data, input_labels=None, dict_args=None, is_test=False):
    feed_dict = super(DaeMemModel, self).get_feed_dict(input_data, input_labels, dict_args, is_test)
    if self.params.memristor_type == "rram":
      mem_std_eps = self.params.rand_state.uniform(low=-1.0, high=1.0,
         size=(self.params.batch_size, self.module.num_latent)).astype(np.float32)
    else:
      mem_std_eps = self.params.rand_state.standard_normal((self.params.batch_size,
         self.module.num_latent)).astype(np.float32)
    feed_dict[self.memristor_std_eps] = mem_std_eps
    return feed_dict
