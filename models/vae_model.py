import numpy as np
import tensorflow as tf
import utils.plot_functions as pf
import utils.data_processing as dp
import utils.entropy_functions as ef
from models.ae_model import AeModel
from modules.vae_module import VaeModule

class VaeModel(AeModel):
  def __init__(self):
    """
    Variational Autoencoder using Mixture of Gaussians
    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
    arXiv preprint arXiv:1312.6114 (2013).
    """
    super(VaeModel, self).__init__()
    self.vector_inputs = True

  def build_module(self, input_node):
    module = VaeModule(input_node, self.params.output_channels, self.sparse_mult,
      self.decay_mult, self.kld_mult, self.act_funcs, self.dropout_keep_probs,
      self.params.tie_decoder_weights, self.params.noise_level, self.params.recon_loss_type)
    return module

  def build_graph_from_input(self, input_node):
    """Build the TensorFlow graph object"""
    with tf.device(self.params.device):
      with self.graph.as_default():
        with tf.name_scope("auto_placeholders") as scope:
          self.sparse_mult = tf.placeholder(tf.float32, shape=(), name="sparse_mult")
          self.kld_mult = tf.placeholder(tf.float32, shape=(), name="kld_mult")
    super(VaeModel, self).build_graph_from_input(input_node)

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
    update_dict = super(VaeModel, self).generate_update_dict(input_data, input_labels, batch_step)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    latent_loss =  tf.get_default_session().run(self.module.loss_dict["latent_loss"], feed_dict)
    stat_dict = {"latent_loss":latent_loss}
    update_dict.update(stat_dict)
    return update_dict

  def generate_plots(self, input_data, input_labels=None):
    """
    Plot weights, reconstruction, and gradients
    Inputs:
      input_data: data object containing the current image batch
      input_labels: data object containing the current label batch
    """
    super(VaeModel, self).generate_plots(input_data, input_labels)
    feed_dict = self.get_feed_dict(input_data, input_labels)
    eval_list = [self.global_step, self.module.w_enc_std, self.module.b_enc_std]
    eval_out = tf.get_default_session().run(eval_list, feed_dict)
    current_step = str(eval_out[0])
    w_enc_std, b_enc_std = eval_out[1:]
    w_enc_std_norm = np.linalg.norm(w_enc_std, axis=0, keepdims=False)
    filename_suffix = "_v"+self.params.version+"_"+current_step.zfill(5)+".png"
    #TODO histogram with large bins is broken
    #fig = pf.plot_activity_hist(b_enc_std, title="Encoding Bias Std Histogram",
    #  save_filename=(self.disp_dir+"b_enc_std_hist_v"+self.version+"-"
    #  +current_step.zfill(5)+".png"))
    latent_layer = self.module.num_encoder_layers-1
    fig = pf.plot_activity_hist(w_enc_std,
      title="Activity Encoder "+str(latent_layer)+" Std Histogram",
      save_filename=(self.params.disp_dir+"act_enc_"+str(latent_layer)
      +"_std_hist"+filename_suffix))
    fig = pf.plot_bar(w_enc_std_norm, num_xticks=5,
      title="w_enc_"+str(latent_layer)+"_std l2 norm", xlabel="Basis Index", ylabel="L2 Norm",
      save_filename=(self.params.disp_dir+"w_enc_"+str(latent_layer)+"_std_norm"
      +filename_suffix))
