import os
import numpy as np
import tensorflow as tf
from analysis.base_analyzer import Analyzer
import utils.data_processing as dp

class SaAnalyzer(Analyzer):
  def __init__(self):
    Analyzer.__init__(self)
    self.var_names = [
      "weights/w_enc:0",
      "inference/activity:0"]

  def add_pre_init_ops_to_graph(self):
    super(SaAnalyzer, self).add_pre_init_ops_to_graph()
    self.add_a_deriv_ops_to_graph()

  def add_a_deriv_ops_to_graph(self):
    with tf.device(self.model.device):
      with self.model.graph.as_default():
        self.model.ax_grad = tf.gradients(tf.slice(self.model.get_encodings(), [0, 0], [-1, 1]),
          self.model.x)[0]

  def run_analysis(self, images, save_info=""):
    super(SaAnalyzer, self).run_analysis(images, save_info)
    if self.analysis_params.do_evals:
      self.evals = self.eval_analysis(images, self.var_names, save_info)
    if self.do_basis_analysis:
      self.bf_stats = self.basis_analysis(self.evals["weights/w_enc:0"], save_info)
    if self.do_atas:
      self.atas, self.atcs = self.ata_analysis(images, self.evals["inference/activity:0"],
        save_info)
      self.noise_activity, self.noise_atas, self.noise_atcs = self.run_noise_analysis(save_info)
    if self.analysis_params.do_orientation_analysis:
      self.ot_grating_responses, self.co_grating_responses = self.grating_analysis(self.bf_stats,
        save_info)
    if self.do_adversaries:
      self.adversarial_images, self.adversarial_recons, mses = self.adversary_analysis(images,
        input_id=self.analysis_params.adversarial_input_id,
        target_id=self.analysis_params.adversarial_target_id,
        eps=self.analysis_params.adversarial_eps,
        num_steps=self.analysis_params.adversarial_num_steps,
        save_info=save_info)
      self.adversarial_input_target_mses = mses["input_target_mse"]
      self.adversarial_input_recon_mses = mses["input_recon_mses"]
      self.adversarial_input_adv_mses = mses["input_adv_mses"]
      self.adversarial_target_recon_mses = mses["target_recon_mses"]
      self.adversarial_target_adv_mses = mses["target_adv_mses"]
      self.adversarial_adv_recon_mses = mses["adv_recon_mses"]
