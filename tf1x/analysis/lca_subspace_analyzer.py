import os

import numpy as np
import tensorflow as tf

from DeepSparseCoding.tf1x.analysis.lca_analyzer import LcaAnalyzer
import DeepSparseCoding.tf1x.utils.data_processing as dp

class LcaSubspaceAnalyzer(LcaAnalyzer):
  def __init__(self):
    super(LcaSubspaceAnalyzer, self).__init__()
    self.var_names = ["lca_subspace/weights/w:0", "inference/activity:0"]

  def run_analysis(self, images, labels=None, save_info=""):
    super(LcaAnalyzer, self).run_analysis(images, labels, save_info=save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.analysis_params.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["lca_subspace/weights/w:0"], save_info)
    if self.analysis_params.do_atas:
      self.atas, self.atcs = self.ata_analysis(
        images[:int(self.analysis_params.num_ata_images), ...],
        self.evals["inference/activity:0"][:int(self.analysis_params.num_ata_images), ...],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_inference:
      self.inference_stats = self.inference_analysis(images, save_info,
        self.analysis_params.num_inference_images, self.analysis_params.num_inference_steps)
    if self.analysis_params.do_orientation_analysis:
      if not self.analysis_params.do_basis_analysis:
        try:
          self.load_basis_stats(save_info)
        except FileNotFoundError:
          assert False, (
          "Basis analysis must have been previously run, or do_basis_analysis must be True.")
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
    if self.analysis_params.do_recon_adversaries:
      self.recon_adversary_analysis(images,
        labels=labels, batch_size=self.analysis_params.eval_batch_size,
        input_id=self.analysis_params.adversarial_input_id,
        target_method=self.analysis_params.adversarial_target_method,
        target_id=self.analysis_params.adversarial_target_id,
        save_info=save_info)
    elif self.analysis_params.do_neuron_visualization:
      self.neuron_visualization_analysis(save_info=save_info)

  def add_pre_init_ops_to_graph(self):
    super(LcaSubspaceAnalyzer, self).add_pre_init_ops_to_graph()
    if self.analysis_params.do_group_recons:
      with tf.device(self.model_params.device):
        with self.model.graph.as_default():
          with tf.compat.v1.variable_scope("placeholders") as scope:
            self.sigmas = tf.compat.v1.placeholder(tf.float32, [None, self.model_params.num_groups])
            self.zs = tf.compat.v1.placeholder(tf.float32, [None, self.model_params.num_groups,
              self.model_params.num_neurons_per_group])
          self.group_recons = self.model.compute_recon_from_group(self.sigmas, self.zs)

  def compute_pooled_activations(self, images):
    """
    Computes the 2nd layer output code for a set of images.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config, graph=self.model.graph) as sess:
      feed_dict = self.model.get_feed_dict(images)
      sess.run(self.model.init_op, feed_dict)
      self.model.load_full_model(sess, self.model_params.cp_loc)
      activations = sess.run(self.model.group_activity, feed_dict)
    return np.squeeze(activations)
